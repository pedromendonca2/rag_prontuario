import asyncio
import json
import uuid
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aio_pika
from chat_memoria import HistoricoMemoria
import config

app = FastAPI(title="RAG Prontuário Médico API")

RABBITMQ_URL = "amqp://guest:guest@localhost/"

# Gerenciamento de histórico em memória (para a PoC)
sessoes_historico: Dict[str, HistoricoMemoria] = {}

class ChatRequest(BaseModel):
    session_id: str
    pergunta: str

class ResumoRequest(BaseModel):
    session_id: str

class RpcClient:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.futures = {}

    async def connect(self):
        self.connection = await aio_pika.connect_robust(RABBITMQ_URL)
        self.channel = await self.connection.channel()
        self.callback_queue = await self.channel.declare_queue(exclusive=True)
        await self.callback_queue.consume(self.on_response, no_ack=True)

    def on_response(self, message: aio_pika.IncomingMessage):
        if message.correlation_id in self.futures:
            future = self.futures.pop(message.correlation_id)
            future.set_result(message.body.decode())

    async def call(self, queue_name: str, payload: dict) -> dict:
        correlation_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.futures[correlation_id] = future

        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode(),
                correlation_id=correlation_id,
                reply_to=self.callback_queue.name,
            ),
            routing_key=queue_name,
        )
        # Aguarda a resposta (síncrono para o cliente HTTP, assíncrono para o FastAPI)
        response_json = await future
        return json.loads(response_json)

rpc_client = RpcClient()

@app.on_event("startup")
async def startup_event():
    await rpc_client.connect()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.session_id not in sessoes_historico:
        sessoes_historico[request.session_id] = HistoricoMemoria()
    
    historico = sessoes_historico[request.session_id]
    
    # Prepara payload para o worker
    payload = {
        "tipo": "chat",
        "pergunta": request.pergunta,
        "historico": historico.listar()
    }
    
    try:
        response = await rpc_client.call("rag_tasks_queue", payload)
        if "erro" in response:
            raise HTTPException(status_code=500, detail=response["erro"])
            
        resposta_texto = response.get("resposta", "")
        
        # Atualiza histórico se houve sucesso
        historico.adicionar("user", request.pergunta)
        historico.adicionar("assistant", resposta_texto)
        
        return {"resposta": resposta_texto}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/resumo")
async def resumo(request: ResumoRequest):
    payload = {
        "tipo": "resumo"
    }
    
    try:
        response = await rpc_client.call("rag_tasks_queue", payload)
        if "erro" in response:
            raise HTTPException(status_code=500, detail=response["erro"])
            
        return {"resumo": response.get("resposta", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

