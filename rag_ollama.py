"""
Módulo de Integração com Ollama e Engenharia de Prompts Clínicos.

Responsável pela comunicação HTTP com o daemon Ollama (hospedado na máquina remota com GPU)
e pela montagem de prompts estritamente concebidos para evitar alucinações clínicas,
garantindo que o modelo de linguagem atue estritamente como relator dos fatos registrados.
"""
import json
import time
from datetime import datetime
import requests

import config


def consultar_ollama(prompt: str, modelo=None, temperatura: float = 0.3, max_tokens: int = 1024, num_ctx: int = None) -> str:
    """
    Envia uma requisição de inferência ao servidor Ollama via HTTP REST (/api/generate).

    Possui política de retentativa em caso de instabilidades pontuais de rede/VPN e
    timeouts configurados especificamente para tarefas de geração de texto clínico.

    Args:
        prompt (str): Prompt textual completo contendo contexto, regras e pergunta.
        modelo (str, opcional): Nome do modelo (padrão: config.MODELO_LLM, ex: 'llama3:latest').
        temperatura (float, opcional): Grau de aleatoriedade da geração (0.1 para resumos factuais, 0.3 para chat).
        max_tokens (int, opcional): Limite máximo de tokens preditos na resposta.
        num_ctx (int, opcional): Janela máxima de contexto em tokens (padrão: 4096).

    Returns:
        str: Texto gerado pelo modelo ou mensagem explicativa em caso de erro.
    """
    modelo = modelo or config.MODELO_LLM
    options = {
        "temperature": temperatura,
        "num_predict": max_tokens,
        "num_ctx": min(num_ctx or config.OLLAMA_NUM_CTX, 4096),
        "num_gpu": config.OLLAMA_NUM_GPU,
    }

    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    
    tentativas = 2
    for tentativa in range(1, tentativas + 1):
        try:
            # Timeout: 5 segundos para conexão e 300 segundos para resposta/inferência do modelo
            resposta = requests.post(config.OLLAMA_URL, json=payload, timeout=(5, 300))
            resposta.raise_for_status()
            return resposta.json()["response"].strip()
        except requests.exceptions.HTTPError as e:
            erro_msg = resposta.text if 'resposta' in locals() else str(e)
            return f"[ERRO] Falha na API do Ollama: {erro_msg}. Você já baixou o modelo rodando 'ollama pull {modelo}'?"
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            if tentativa < tentativas:
                time.sleep(1)
                continue
            return (
                f"[ERRO] Não foi possível conectar ao Ollama em {config.OLLAMA_URL}. "
                "Verifique se o servidor de GPU está ligado, se a VPN está ativa e se o comando "
                "'nohup env OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=/home/shared/git/Development/llama/models /home/shared/git/Development/llama/bin/ollama serve > ~/ollama.log 2>&1 &' "
                "está em execução no servidor remoto."
            )
        except requests.exceptions.ReadTimeout:
            return f"[ERRO] O modelo Ollama excedeu o tempo limite de resposta (5 minutos)."
        except Exception as e:
            return f"[ERRO] Falha ao consultar Ollama: {e}"


def formatar_chunks_para_contexto(chunks: list) -> str:
    """
    Formata tuplas de chunks recuperados pelo ChromaDB em texto limpo com data e tipo.

    Args:
        chunks (list[tuple]): Lista de tuplas (documento, metadados, distancia).

    Returns:
        str: String consolidada separada por quebras duplas de linha.
    """
    linhas = []
    for doc, meta, dist in chunks:
        data = meta.get("data", "")
        tipo = meta.get("tipo_evento", "")
        linhas.append(f"[{data}] ({tipo}) {doc}")
    return "\n\n".join(linhas)


def formatar_historico_para_prompt(historico) -> str:
    """
    Converte o histórico da sessão em diálogo textual no formato 'Medico: ... / Assistente: ...'.

    Args:
        historico (HistoricoMemoria): Instância com as mensagens recentes.

    Returns:
        str: String formatada com os turnos anteriores.
    """
    if not historico.listar():
        return "Nenhum histórico anterior."
    linhas = []
    for turno in historico.listar():
        papel = "Médico" if turno["role"] == "user" else "Assistente"
        linhas.append(f"{papel}: {turno['content']}")
    return "\n".join(linhas)


def montar_prompt_resumo(contexto: str) -> str:
    """
    Monta o prompt restritivo do Resumo Clínico Estruturado em 7 seções obrigatórias.

    Impõe regras estritas contra alucinação e orienta o modelo a atuar estritamente
    como assistente clínico extraindo dados de forma concisa.

    Args:
        contexto (str): Texto com todo o histórico de evoluções médicas do paciente.

    Returns:
        str: Prompt formatado para o Ollama.
    """
    return (
        "Abaixo está o prontuário clínico completo de um paciente com todo o seu histórico em ordem cronológica:\n\n"
        f"--- INÍCIO DO PRONTUÁRIO ---\n{contexto}\n--- FIM DO PRONTUÁRIO ---\n\n"
        "Com base EXCLUSIVAMENTE em todos os registros acima, atue como assistente clínico e elabore o Resumo Clínico Estruturado.\n"
        "REGRAS:\n"
        "- Preencha OBRIGATORIAMENTE todas as 7 seções abaixo com os fatos e dados reais contidos nos registros (idade, valores de PA, remédios, exames, etc.).\n"
        "- Não faça suposições, interpretações ou predições. Não invente dados não mencionados.\n"
        "- Seja conciso e direto em cada tópico.\n\n"
        "ESTRUTURA OBRIGATÓRIA (formato Markdown exato):\n"
        "**1. Dados do paciente**\n"
        "* Idade: \n"
        "* Condição inicial (admissão): \n\n"
        "**2. Resumo da evolução temporal (linha do tempo)**\n"
        "* \n\n"
        "**3. Medicações utilizadas e histórico de doses**\n"
        "* \n\n"
        "**4. Eventos adversos e intercorrências**\n"
        "* \n\n"
        "**5. Resultados de exames laboratoriais e complementares**\n"
        "* \n\n"
        "**6. Controle pressórico atual vs inicial**\n"
        "* Inicial: \n"
        "* Atual/Final: \n\n"
        "**7. Conduta atual e plano de seguimento**\n"
        "* \n\n"
        "Resumo Clínico Estruturado:"
    )


def montar_prompt_rag(pergunta: str, chunks: list, historico) -> str:
    """
    Monta o prompt de perguntas e respostas para o Chat Clínico.

    Aplica 6 diretrizes essenciais desenvolvidas para o ambiente médico:
    1. Fluidez e Profissionalismo Clínico (evitar respostas abruptas ou secas).
    2. Objetividade Factual (fundamentar com dados reais do prontuário).
    3. Perguntas Múltiplas (responder a cada item questionado pelo médico).
    4. Sem Repetição (ir direto ao ponto solicitado sem copiar histórico).
    5. Terminologia e Sinônimos Clínicos (ex: cefaleia = dor de cabeça).
    6. Evolução e Linha do Tempo (consultar os registros mais recentes para valores finais).

    Args:
        pergunta (str): Pergunta do médico.
        chunks (list): Trechos mais relevantes do prontuário recuperados pelo ChromaDB.
        historico (HistoricoMemoria): Histórico da conversa recente.

    Returns:
        str: Prompt pronto para envio à LLM.
    """
    contexto = formatar_chunks_para_contexto(chunks)
    historico_str = formatar_historico_para_prompt(historico)

    return (
        "Você é um assistente clínico especializado em prontuários médicos. "
        "Sua função é responder à pergunta do médico baseando-se com rigor factual EXCLUSIVAMENTE nos registros do paciente e no histórico da conversa.\n\n"
        "DIRETRIZES DE RESPOSTA:\n"
        "1. FLUIDEZ E PROFISSIONALISMO CLÍNICO: Responda em tom natural, claro e profissional. Nunca comece abruptamente com números ou siglas isoladas (ex: evite iniciar seco com 'PA 122x74 mmHg'). Elabore respostas contextualizadas em frases completas e gramaticalmente coesas. Adequar sempre o pronome e gênero ao paciente identificado nos registros (ex: 'o paciente João' / 'a paciente Maria').\n"
        "2. OBJETIVIDADE FACTUAL: Nunca responda com frases vagas sem embasamento (como apenas 'Está tudo bem'). Fundamente sempre com os dados reais dos registros (diagnóstico, PA mais recente, queixas, medicações ativas).\n"
        "3. PERGUNTAS MÚLTIPLAS: Se o médico fizer várias perguntas na mesma mensagem (ex: idade, taxas de exames, conduta), responda CADA UMA DELAS ponto a ponto. Se algum dado específico não constar no contexto, indique pontualmente qual dado não foi encontrado (ex: 'Taxa de Vitamina D: 35 ng/mL. Idade: não consta nos registros.'), sem descartar os outros fatos encontrados.\n"
        "4. SEM REPETIÇÃO: Não repita respostas anteriores nem copie trechos do histórico no início da sua mensagem. Vá direto ao assunto solicitado.\n"
        "5. TERMINOLOGIA E SINÔNIMOS: Reconheça termos clínicos e sinônimos médicos equivalentes (por exemplo: escotomas visuais são alterações visuais; cefaleia é dor de cabeça; pirose é azia/queimação). Não invente diagnósticos ou dados não documentados.\n"
        "6. EVOLUÇÃO E LINHA DO TEMPO: Os registros do prontuário estão dispostos em ordem cronológica. Para perguntas sobre totais acumulados ou valores finais/atuais (ex: perda total de peso acumulada ao final do acompanhamento, pressão arterial atual, dose atual de medicação), consulte SEMPRE o registro cronológico mais recente contido no prontuário.\n\n"
        "Registros do Prontuário:\n"
        f"{contexto}\n\n"
        "Histórico recente da conversa:\n"
        f"{historico_str}\n\n"
        "Pergunta do médico:\n"
        f"{pergunta}\n\n"
        "Resposta clínica:"
    )


def gerar_resumo_clinico(indexador, paciente_id=None, modelo=None) -> str:
    """
    Orquestra a montagem do contexto histórico e gera o Resumo Clínico Estruturado via LLM.

    Recupera todos os registros cadastrados para a matrícula especificada, ordena-os
    cronologicamente para que o modelo visualize a evolução completa da saúde do paciente
    e aciona o Ollama com temperatura 0.1 e janela de contexto de 4096 tokens.

    Args:
        indexador (IndexadorVectorial): Instância ativa de acesso ao ChromaDB.
        paciente_id (str, opcional): Matrícula do paciente.
        modelo (str, opcional): Modelo LLM a ser utilizado.

    Returns:
        str: Resumo Clínico Estruturado formatado em Markdown.
    """
    paciente_str = str(paciente_id).strip() if paciente_id else None
    print(f"[rag] Gerando resumo clínico para paciente: {paciente_str or 'todos'}...")

    where_filter = {"paciente_id": paciente_str} if paciente_str else None
    if where_filter:
        dados = indexador.colecao.get(where=where_filter)
    else:
        dados = indexador.colecao.get()

    if not dados or not dados.get("documents"):
        return f"Ainda não há registros no prontuário do paciente (matrícula: {paciente_str or 'não informada'}) para gerar o resumo clínico."

    def parse_data_meta(meta):
        d_str = meta.get("data", "")
        try:
            return datetime.strptime(d_str.split()[0], "%d/%m/%Y")
        except Exception:
            return datetime.min

    itens = list(zip(dados["ids"], dados["documents"], dados["metadatas"]))
    itens.sort(key=lambda x: parse_data_meta(x[2]))

    docs_formatados = []
    for _, doc, meta in itens:
        linha = f"[{meta.get('data', '')} - {meta.get('tipo_evento', '')}] {doc}"
        docs_formatados.append(linha)

    chunks_texto = "\n\n".join(docs_formatados)
    print(f"[rag] Contexto montado com {len(docs_formatados)} registros (~{len(chunks_texto)} chars)")

    prompt = montar_prompt_resumo(chunks_texto)
    print("[rag] Enviando para Ollama com num_ctx=4096 e temperatura=0.1...")
    resposta = consultar_ollama(prompt, modelo=modelo, temperatura=0.1, max_tokens=1000, num_ctx=4096)
    return resposta
