"""
Módulo da API REST Web (FastAPI) e Cliente RPC Assíncrono com RabbitMQ.

Este módulo atua como a camada de entrada HTTP para as aplicações cliente
(como o frontend Angular 'plugin-ease-stream').

Características Principais:
1. Arquitetura Desacoplada (RPC): O FastAPI recebe requisições HTTP, publica uma mensagem
   na fila 'rag_tasks_queue' com um Correlation ID exclusivo e aguarda a resposta assincronamente.
   Isso impede que o servidor web bloqueie suas threads com inferências de IA.
2. Sanitização HTML: Remove tags de editores WYSIWYG antes de salvar ou vetorizar.
3. Isolamento Multi-Paciente: Cada requisição exige ou referencia a matrícula do paciente ('session_id' ou 'paciente_id').
4. Suporte a CORS: Pré-configurado para integração com o Angular em http://localhost:4300.
"""
import asyncio
import json
import re
import uuid
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import aio_pika
from chat_memoria import HistoricoMemoria
import config

app = FastAPI(
    title="RAG Prontuário Médico API",
    description="API REST de alto desempenho para consulta semântica, geração de resumos clínicos e ingestão de prontuários via RAG.",
    version="2.0.0",
)

# Configuração de CORS para permitir consumo pelo frontend Angular local e ambientes homologados
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4300", "http://localhost:4203", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RABBITMQ_URL = config.RABBITMQ_URL

# Gerenciamento de histórico de diálogo em memória indexado por matrícula de paciente
sessoes_historico: Dict[str, HistoricoMemoria] = {}


# --- MODELOS PYDANTIC DE VALIDAÇÃO DE DADOS ---

class ChatRequest(BaseModel):
    """Payload para envio de perguntas clínicas ao assistente RAG."""
    session_id: str = Field(..., description="Matrícula ou identificador único do paciente (obrigatória)")
    pergunta: str = Field(..., description="Pergunta clínica formulada pelo profissional de saúde")


class ResumoRequest(BaseModel):
    """Payload para solicitação de Resumo Clínico Estruturado."""
    session_id: str = Field(..., description="Matrícula ou identificador único do paciente (obrigatória)")


class LimparChatRequest(BaseModel):
    """Payload para reset do histórico de diálogo."""
    session_id: str = Field(..., description="Matrícula do paciente a ser limpa")


class AdicionarRegistroRequest(BaseModel):
    """Payload para inclusão dinâmica de nova evolução no prontuário."""
    paciente_id: str = Field(..., description="Matrícula ou identificador único do paciente")
    texto: str = Field(..., description="Conteúdo textual da ficha ou evolução médica")
    data: Optional[str] = Field(None, description="Data no formato DD/MM/AAAA ou DD/MM/AAAA HH:MM")
    tipo_evento: Optional[str] = Field("Evolução Médica", description="Tipo do registro (ex: Consulta, Retorno, Exame)")
    profissional: Optional[str] = Field(None, description="Nome ou CRM do profissional de saúde")


def sanitizar_html(texto: str) -> str:
    """
    Higieniza textos oriundos de editores ricos (Rich Text / WYSIWYG).

    Converte quebras <br> e parágrafos </p> em quebras de linha reais
    e elimina tags HTML, evitando poluir os embeddings vetoriais com marcações visuais.

    Args:
        texto (str): Texto contendo potenciais tags HTML.

    Returns:
        str: Texto limpo em texto puro.
    """
    if not texto:
        return ""
    # Converte quebras <br>, </p> em quebra de linha
    t = re.sub(r'(?i)<br\s*/?>', '\n', texto)
    t = re.sub(r'(?i)</p>', '\n', t)
    # Remove qualquer outra tag html remanescente
    t = re.sub(r'<[^>]+>', '', t)
    linhas = [l.strip() for l in t.splitlines() if l.strip()]
    return "\n".join(linhas)


# --- CLIENTE RPC AMQP (RABBITMQ) ---

class RpcClient:
    """
    Cliente RPC assíncrono sobre RabbitMQ utilizando a biblioteca aio_pika.

    Cria uma fila exclusiva de resposta ('reply_to') e mapeia respostas
    através de Correlation IDs únicos para suportar alta concorrência sem bloqueio.
    """

    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.futures = {}

    async def connect(self):
        """Estabelece a conexão com o RabbitMQ e declara a fila exclusiva de retorno."""
        self.connection = await aio_pika.connect_robust(RABBITMQ_URL)
        self.channel = await self.connection.channel()
        self.callback_queue = await self.channel.declare_queue(exclusive=True)
        await self.callback_queue.consume(self.on_response, no_ack=True)

    async def ensure_connected(self):
        """Reconecta automaticamente caso o canal ou conexão tenham sido interrompidos."""
        if (
            self.connection is None
            or self.connection.is_closed
            or self.channel is None
            or self.channel.is_closed
            or self.callback_queue is None
        ):
            await self.connect()

    def on_response(self, message: aio_pika.IncomingMessage):
        """Callback acionado quando o worker publica a resposta na fila de retorno."""
        if message.correlation_id in self.futures:
            future = self.futures.pop(message.correlation_id)
            if not future.done():
                future.set_result(message.body.decode())

    async def call(self, queue_name: str, payload: dict, timeout: float = 360.0) -> dict:
        """
        Publica uma mensagem na fila alvo e aguarda a resposta assincronamente.

        Args:
            queue_name (str): Nome da fila consumida pelo worker (ex: 'rag_tasks_queue').
            payload (dict): Dicionário com os dados da tarefa a ser executada.
            timeout (float): Tempo limite em segundos para aguardar a resposta.

        Returns:
            dict: Resposta retornada pelo worker.

        Raises:
            HTTPException (504): Se o tempo limite for atingido.
        """
        await self.ensure_connected()

        correlation_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.futures[correlation_id] = future

        try:
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    correlation_id=correlation_id,
                    reply_to=self.callback_queue.name,
                ),
                routing_key=queue_name,
            )
            response_json = await asyncio.wait_for(future, timeout=timeout)
            return json.loads(response_json)
        except asyncio.TimeoutError:
            self.futures.pop(correlation_id, None)
            raise HTTPException(
                status_code=504,
                detail=f"Tempo limite de {int(timeout)}s excedido aguardando resposta do worker RAG.",
            )
        except Exception as e:
            self.futures.pop(correlation_id, None)
            raise e


rpc_client = RpcClient()


@app.on_event("startup")
async def startup_event():
    """Conecta ao broker RabbitMQ na inicialização da aplicação FastAPI."""
    await rpc_client.connect()


# --- ENDPOINTS DA API REST ---

@app.post("/api/prontuario", status_code=status.HTTP_201_CREATED, summary="Adicionar nova evolução médica")
async def adicionar_prontuario(request: AdicionarRegistroRequest):
    """
    Recebe uma nova evolução clínica, sanitiza o conteúdo e despacha para o worker indexar no ChromaDB.
    """
    paciente_id = request.paciente_id.strip() if request.paciente_id else ""
    if not paciente_id:
        raise HTTPException(status_code=400, detail="A matrícula do paciente ('paciente_id') é obrigatória.")

    texto_limpo = sanitizar_html(request.texto)
    if not texto_limpo:
        raise HTTPException(status_code=400, detail="O texto da evolução ('texto') não pode ser vazio.")

    payload = {
        "tipo": "adicionar_registro",
        "paciente_id": paciente_id,
        "texto": texto_limpo,
        "data": request.data.strip() if request.data else None,
        "tipo_evento": request.tipo_evento.strip() if request.tipo_evento else "Evolução Médica",
        "profissional": request.profissional.strip() if request.profissional else None,
    }

    try:
        response = await rpc_client.call("rag_tasks_queue", payload, timeout=60.0)
        if "erro" in response:
            raise HTTPException(status_code=500, detail=response["erro"])

        return {
            "status": "sucesso",
            "mensagem": "Evolução adicionada ao prontuário com sucesso.",
            "registro": response.get("registro"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prontuario/{paciente_id}", summary="Listar prontuário de um paciente")
async def listar_prontuario(paciente_id: str):
    """
    Recupera todas as evoluções clínicas cadastradas para o paciente informado, ordenadas cronologicamente.
    """
    pid = paciente_id.strip() if paciente_id else ""
    if not pid:
        raise HTTPException(status_code=400, detail="A matrícula do paciente é obrigatória.")

    payload = {
        "tipo": "listar_prontuario",
        "paciente_id": pid,
    }

    try:
        response = await rpc_client.call("rag_tasks_queue", payload, timeout=30.0)
        if "erro" in response:
            raise HTTPException(status_code=500, detail=response["erro"])

        registros = response.get("registros", [])
        return {
            "status": "sucesso",
            "paciente_id": pid,
            "total": len(registros),
            "registros": registros,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", summary="Conversar com o assistente RAG sobre o prontuário")
async def chat(request: ChatRequest):
    """
    Processa uma pergunta clínica sobre o histórico do paciente utilizando RAG vetorial e LLM.
    """
    paciente_id = request.session_id.strip() if request.session_id else ""
    if not paciente_id:
        raise HTTPException(status_code=400, detail="A matrícula do paciente ('session_id') é obrigatória.")

    pergunta_limpa = request.pergunta.strip() if request.pergunta else ""
    if not pergunta_limpa:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")
    
    if len(pergunta_limpa) > 4000:
        raise HTTPException(status_code=400, detail="A pergunta excede o tamanho máximo permitido de 4000 caracteres.")

    if paciente_id not in sessoes_historico:
        sessoes_historico[paciente_id] = HistoricoMemoria()
    
    historico = sessoes_historico[paciente_id]
    
    payload = {
        "tipo": "chat",
        "pergunta": pergunta_limpa,
        "paciente_id": paciente_id,
        "historico": historico.listar(),
    }
    
    try:
        response = await rpc_client.call("rag_tasks_queue", payload)
        if "erro" in response:
            raise HTTPException(status_code=500, detail=response["erro"])
            
        resposta_texto = response.get("resposta", "")
        
        # Registra a interação no histórico da sessão
        historico.adicionar("user", pergunta_limpa)
        historico.adicionar("assistant", resposta_texto)
        
        return {"resposta": resposta_texto}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/limpar", summary="Limpar histórico da sessão de chat")
@app.delete("/api/chat/{session_id}", summary="Excluir histórico de chat por matrícula")
async def limpar_chat(session_id: Optional[str] = None, request: Optional[LimparChatRequest] = None):
    """
    Esvazia a memória de turnos conversacionais associada à matrícula informada.
    """
    sid = session_id or (request.session_id if request else None)
    if not sid or not sid.strip():
        raise HTTPException(status_code=400, detail="A matrícula do paciente ('session_id') é obrigatória.")

    sid_limpo = sid.strip()
    if sid_limpo in sessoes_historico:
        del sessoes_historico[sid_limpo]
    return {"status": "ok", "mensagem": f"Histórico da sessão '{sid_limpo}' limpo com sucesso."}


@app.post("/api/resumo", summary="Gerar Resumo Clínico Estruturado")
async def resumo(request: ResumoRequest):
    """
    Gera o Resumo Clínico Estruturado (7 seções padronizadas) de todo o histórico do paciente.
    """
    paciente_id = request.session_id.strip() if request.session_id else ""
    if not paciente_id:
        raise HTTPException(status_code=400, detail="A matrícula do paciente ('session_id') é obrigatória.")

    payload = {
        "tipo": "resumo",
        "paciente_id": paciente_id,
    }
    
    try:
        # Timeout estendido para 900s (15 min) para suportar leitura de prontuários com 100+ evoluções
        response = await rpc_client.call("rag_tasks_queue", payload, timeout=900.0)
        if "erro" in response:
            raise HTTPException(status_code=500, detail=response["erro"])
            
        return {"resumo": response.get("resposta", "")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
