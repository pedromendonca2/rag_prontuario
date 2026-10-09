import asyncio
import json
import logging
import aio_pika

import config
from indexador import IndexadorVectorial
from rag_ollama import consultar_ollama, montar_prompt_rag, gerar_resumo_clinico
from chat_memoria import HistoricoMemoria

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RABBITMQ_URL = "amqp://guest:guest@localhost/"
QUEUE_NAME = "rag_tasks_queue"

# Inicializa recursos de IA uma única vez na inicialização do worker
logger.info("Inicializando modelo de busca vetorial...")
indexador = IndexadorVectorial()
logger.info("Indexador carregado.")

def processar_chat_sincrono(pergunta: str, historico_dados: list) -> str:
    # Reconstrói objeto de histórico a partir da mensagem JSON
    historico = HistoricoMemoria()
    historico._turnos = historico_dados

    chunks = indexador.buscar_por_texto(pergunta)
    if not chunks:
        return "Não encontrei contexto relevante nos prontuários para responder a essa pergunta."

    prompt = montar_prompt_rag(pergunta, chunks, historico)
    logger.info("Consultando LLM para Chat...")
    resposta = consultar_ollama(prompt, modelo=config.MODELO_LLM)
    return resposta

def processar_resumo_sincrono() -> str:
    logger.info("Gerando Resumo via LLM...")
    resposta = gerar_resumo_clinico(indexador, modelo=config.MODELO_LLM)
    return resposta

async def process_message(message: aio_pika.IncomingMessage):
    async with message.process():
        try:
            payload = json.loads(message.body.decode())
            tipo = payload.get("tipo")
            logger.info(f"Processando task do tipo: {tipo}")
            
            resposta_texto = ""
            
            if tipo == "chat":
                pergunta = payload.get("pergunta")
                historico = payload.get("historico", [])
                # Executa processamento bloqueante (Chroma/Ollama) em uma thread separada
                resposta_texto = await asyncio.to_thread(processar_chat_sincrono, pergunta, historico)
            elif tipo == "resumo":
                resposta_texto = await asyncio.to_thread(processar_resumo_sincrono)
            else:
                raise ValueError("Tipo de tarefa desconhecido")

            response_payload = {"resposta": resposta_texto}
        except Exception as e:
            logger.error(f"Erro ao processar: {e}")
            response_payload = {"erro": str(e)}

        # Responder para o Correlation ID via Fila de Resposta (reply_to)
        if message.reply_to:
            try:
                connection = await aio_pika.connect_robust(RABBITMQ_URL)
                channel = await connection.channel()
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(response_payload).encode(),
                        correlation_id=message.correlation_id,
                    ),
                    routing_key=message.reply_to,
                )
                await connection.close()
            except Exception as e:
                logger.error(f"Falha ao enviar resposta RPC: {e}")

async def main():
    logger.info("Conectando ao RabbitMQ...")
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()

    # Garantir que só pega 1 task pesada por vez (prefetch)
    await channel.set_qos(prefetch_count=1)
    
    queue = await channel.declare_queue(QUEUE_NAME, durable=True)
    logger.info(f"Worker iniciado. Aguardando mensagens na fila '{QUEUE_NAME}'...")
    
    await queue.consume(process_message)

    try:
        await asyncio.Future() # Roda para sempre
    finally:
        await connection.close()

if __name__ == "__main__":
    asyncio.run(main())

