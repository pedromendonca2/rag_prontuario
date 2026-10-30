"""
Módulo do Worker de Inteligência Artificial e Processamento Assíncrono AMQP.

Este serviço atua como o consumidor principal da fila 'rag_tasks_queue' no RabbitMQ.
Ele implementa o padrão Remote Procedure Call (RPC):
1. Mantém recursos pesados de IA (SentenceTransformers e ChromaDB) carregados em memória.
2. Consome tarefas pesadas uma a uma (prefetch_count=1) para não sobrecarregar a GPU ou CPU.
3. Executa as tarefas em threads separadas (asyncio.to_thread) para não bloquear o loop de eventos.
4. Publica a resposta diretamente na fila temporária do solicitante ('reply_to') usando o 'correlation_id'.
"""
import asyncio
import json
import logging
import aio_pika
import re
from datetime import datetime

import config
from indexador import IndexadorVectorial
from rag_ollama import consultar_ollama, montar_prompt_rag, gerar_resumo_clinico
from chat_memoria import HistoricoMemoria
from persistencia import salvar_novo_registro_dinamico

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_worker")

RABBITMQ_URL = config.RABBITMQ_URL
QUEUE_NAME = "rag_tasks_queue"

# Inicializa recursos pesados de embeddings e ChromaDB na inicialização do container/processo
logger.info("Inicializando modelo de busca vetorial e conexões com ChromaDB...")
indexador = IndexadorVectorial()
logger.info(f"Indexador carregado com sucesso. Coleção possui {indexador.contar()} documentos.")


def extrair_sub_queries(pergunta: str) -> list[str]:
    """
    Decompõe perguntas clínicas complexas ou compostas em sub-queries menores.

    Permite que perguntas com múltiplos questionamentos (ex: 'Qual a PA? Quais remédios?')
    gerem buscas independentes no banco vetorial, recuperando todos os contextos necessários.

    Args:
        pergunta (str): Pergunta original do médico.

    Returns:
        list[str]: Lista de sub-perguntas com tamanho relevante (> 3 caracteres).
    """
    partes = re.split(r'(?:\d+[\)\.]|\?|\n)', pergunta)
    queries = [p.strip() for p in partes if len(p.strip()) > 3]
    if pergunta not in queries:
        queries.insert(0, pergunta)
    return queries


def recuperar_chunks_contexto(pergunta: str, paciente_id: str = None) -> list:
    """
    Recupera, filtra por paciente, desduplica e ordena cronologicamente os chunks mais relevantes.

    Estratégia clínica empregada:
    1. Multi-query search: busca vetorial paralela para cada aspecto da pergunta.
    2. Âncora de Admissão: injeta forçadamente o primeiro registro histórico do paciente
       para fornecer a linha de base (diagnóstico de entrada, idade, sintomas iniciais).
    3. Seleção por relevância: seleciona os top-N chunks com menor distância semântica.
    4. Ordenação Cronológica: reordena os chunks finais por data (DD/MM/AAAA) para permitir
       que a LLM entenda o fluxo temporal e causal do tratamento.

    Args:
        pergunta (str): Pergunta do médico.
        paciente_id (str, opcional): Matrícula para isolamento dos dados.

    Returns:
        list[tuple]: Lista de tuplas (texto, metadados, distancia) ordenadas cronologicamente.
    """
    queries = extrair_sub_queries(pergunta)
    chunks_dict = {}

    paciente_str = str(paciente_id).strip() if paciente_id else None
    where_filter = {"paciente_id": paciente_str} if paciente_str else None

    # Ajusta o k por sub-query para não estourar o contexto
    k_per_q = max(10, config.TOP_K_BUSCA // len(queries) + 4) if len(queries) > 1 else config.TOP_K_BUSCA

    resultados = indexador.buscar_multiplas_queries(queries, k=k_per_q, where=where_filter)
    for doc, meta, dist in resultados:
        if doc not in chunks_dict or dist < chunks_dict[doc][2]:
            chunks_dict[doc] = (doc, meta, dist)

    # Injeção da âncora cronológica (primeiro registro do paciente)
    try:
        if where_filter:
            registros_paciente = indexador.colecao.get(where=where_filter)
            if registros_paciente and registros_paciente.get("documents"):
                def parse_dt(meta):
                    d_str = meta.get("data", "")
                    try:
                        return datetime.strptime(d_str.split()[0], "%d/%m/%Y")
                    except Exception:
                        return datetime.min

                itens_paciente = list(zip(registros_paciente["documents"], registros_paciente["metadatas"]))
                itens_paciente.sort(key=lambda x: parse_dt(x[1]))
                doc_primeiro, meta_primeiro = itens_paciente[0]
                if doc_primeiro not in chunks_dict:
                    chunks_dict[doc_primeiro] = (doc_primeiro, meta_primeiro, 0.0)
        else:
            registro_admissao = indexador.colecao.get(ids=["1"])
            if registro_admissao and registro_admissao.get("documents"):
                doc_adm = registro_admissao["documents"][0]
                if doc_adm not in chunks_dict:
                    meta_adm = registro_admissao["metadatas"][0] if registro_admissao.get("metadatas") else {}
                    chunks_dict[doc_adm] = (doc_adm, meta_adm, 0.0)
    except Exception as e:
        logger.warning(f"Não foi possível anexar registro inicial como âncora: {e}")

    if not chunks_dict:
        return []

    # Seleciona os 35 trechos semanticamente mais próximos
    top_chunks = sorted(chunks_dict.values(), key=lambda x: x[2])[:35]

    # Ordena cronologicamente por data para a LLM entender a linha do tempo do paciente
    def parse_data(c):
        d_str = c[1].get("data", "")
        try:
            return datetime.strptime(d_str.split()[0], "%d/%m/%Y")
        except Exception:
            return datetime.min

    top_chunks.sort(key=parse_data)
    return top_chunks


def processar_chat_sincrono(pergunta: str, historico_dados: list, paciente_id: str = None) -> str:
    """
    Processa uma consulta interativa de chat médico.

    Args:
        pergunta (str): Pergunta do médico.
        historico_dados (list): Lista de turnos prévios da conversa.
        paciente_id (str, opcional): Matrícula do paciente.

    Returns:
        str: Resposta clínica fundamentada gerada pela LLM.
    """
    historico = HistoricoMemoria()
    historico._turnos = historico_dados

    chunks = recuperar_chunks_contexto(pergunta, paciente_id=paciente_id)

    if not chunks:
        return f"Ainda não há registros no prontuário para o paciente informado (matrícula: {paciente_id or 'não informada'})."

    prompt = montar_prompt_rag(pergunta, chunks, historico)
    logger.info(f"Consultando LLM ({config.MODELO_LLM}) para chat clínico (paciente: {paciente_id})...")
    resposta = consultar_ollama(prompt, modelo=config.MODELO_LLM, temperatura=0.1)
    return resposta


def processar_resumo_sincrono(paciente_id: str = None) -> str:
    """
    Processa a síntese de todo o histórico clínico de um paciente.

    Args:
        paciente_id (str, opcional): Matrícula do paciente.

    Returns:
        str: Resumo Clínico Estruturado em 7 seções Markdown.
    """
    logger.info(f"Gerando Resumo via LLM para paciente: {paciente_id}...")
    resposta = gerar_resumo_clinico(indexador, paciente_id=paciente_id, modelo=config.MODELO_LLM)
    return resposta


def processar_adicionar_registro(payload: dict) -> dict:
    """
    Adiciona dinamicamente uma nova evolução médica no ChromaDB e no JSON de backup.

    Args:
        payload (dict): Contém paciente_id, texto, data, tipo_evento e profissional.

    Returns:
        dict: Registro adicionado e indexado.
    """
    paciente_id = payload.get("paciente_id")
    texto = payload.get("texto")
    data = payload.get("data")
    tipo_evento = payload.get("tipo_evento")
    profissional = payload.get("profissional")

    logger.info(f"Adicionando novo registro no banco vetorial para paciente: {paciente_id}...")
    registro = indexador.adicionar_registro(
        texto=texto,
        paciente_id=paciente_id,
        data=data,
        tipo_evento=tipo_evento,
        profissional=profissional,
    )

    try:
        salvar_novo_registro_dinamico(registro, config.ARQUIVO_DINAMICO)
        logger.info(f"Backup incremental salvo com sucesso em {config.ARQUIVO_DINAMICO}")
    except Exception as e:
        logger.error(f"Erro ao salvar backup dinâmico: {e}")

    return registro


def processar_listar_prontuario(paciente_id: str) -> list:
    """
    Lista todos os registros médicos de um paciente em ordem cronológica.

    Args:
        paciente_id (str): Matrícula do paciente.

    Returns:
        list[dict]: Lista de evoluções médicas.
    """
    logger.info(f"Listando prontuário cronológico para paciente: {paciente_id}...")
    return indexador.listar_registros_paciente(paciente_id)


rpc_channel = None


async def process_message(message: aio_pika.IncomingMessage):
    """
    Callback assíncrono para processamento de mensagens recebidas da fila AMQP.

    Decodifica o payload JSON, despacha a tarefa correspondente para execução em thread,
    e publica o resultado na fila temporária especificada no cabeçalho 'reply_to'.
    """
    async with message.process():
        try:
            payload = json.loads(message.body.decode())
            tipo = payload.get("tipo")
            logger.info(f"Processando tarefa RPC do tipo: '{tipo}'...")
            
            response_payload = {}
            
            if tipo == "chat":
                pergunta = payload.get("pergunta")
                historico = payload.get("historico", [])
                paciente_id = payload.get("paciente_id")
                resposta_texto = await asyncio.to_thread(processar_chat_sincrono, pergunta, historico, paciente_id)
                response_payload = {"resposta": resposta_texto}
            elif tipo == "resumo":
                paciente_id = payload.get("paciente_id")
                resposta_texto = await asyncio.to_thread(processar_resumo_sincrono, paciente_id)
                response_payload = {"resposta": resposta_texto}
            elif tipo == "adicionar_registro":
                novo_reg = await asyncio.to_thread(processar_adicionar_registro, payload)
                response_payload = {"status": "ok", "registro": novo_reg}
            elif tipo == "listar_prontuario":
                paciente_id = payload.get("paciente_id")
                registros = await asyncio.to_thread(processar_listar_prontuario, paciente_id)
                response_payload = {"status": "ok", "registros": registros}
            else:
                raise ValueError(f"Tipo de tarefa desconhecido: '{tipo}'")

        except Exception as e:
            logger.error(f"Erro ao processar mensagem AMQP: {e}", exc_info=True)
            response_payload = {"erro": str(e)}

        # Responde diretamente na fila exclusiva temporária do cliente HTTP (reply_to)
        if message.reply_to and rpc_channel:
            try:
                await rpc_channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(response_payload).encode(),
                        correlation_id=message.correlation_id,
                    ),
                    routing_key=message.reply_to,
                )
                logger.info(f"Resposta RPC publicada com sucesso para {message.reply_to} (Correlation: {message.correlation_id})")
            except Exception as e:
                logger.error(f"Falha ao enviar resposta RPC: {e}")


async def main():
    """
    Ponto de entrada assíncrono do worker.

    Conecta ao RabbitMQ, declara a fila durável e inicia o consumo contínuo.
    """
    global rpc_channel
    logger.info(f"Conectando ao broker RabbitMQ em {RABBITMQ_URL}...")
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    rpc_channel = await connection.channel()

    # Prefetch=1 garante que cada worker só assume uma tarefa pesada de IA por vez
    await rpc_channel.set_qos(prefetch_count=1)
    
    queue = await rpc_channel.declare_queue(QUEUE_NAME, durable=True)
    logger.info(f"Worker ativo e aguardando requisições na fila '{QUEUE_NAME}'...")
    
    await queue.consume(process_message)

    try:
        await asyncio.Future()  # Executa indefinidamente até receber sinal de parada
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
