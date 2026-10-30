"""
Módulo de Configuração Central do RAG Prontuário Médico.

Centraliza o carregamento de variáveis de ambiente a partir do arquivo .env
e define constantes fundamentais para a conexão com o RabbitMQ, ChromaDB,
SentenceTransformers e Ollama.

Variáveis de Ambiente Suportadas:
- MODELO_EMBEDDING: Identificador HuggingFace do modelo de embeddings vetoriais.
- MODELO_LLM: Nome do modelo de linguagem no servidor Ollama (ex: llama3:latest).
- OLLAMA_URL: Endpoint HTTP da API do Ollama (ex: http://192.168.88.2:11434/api/generate).
- OLLAMA_NUM_GPU: Camadas offloaded para a GPU (ex: 22 ou 33).
- OLLAMA_NUM_CTX: Tamanho da janela de contexto de inferência (em tokens).
- RABBITMQ_URL: String de conexão AMQP (ex: amqp://guest:guest@localhost/).
- LIMITE_HISTORICO: Quantidade máxima de turnos conversacionais no buffer de memória.
- TOP_K_BUSCA: Quantidade de documentos mais similares recuperados pelo ChromaDB.
- ARQUIVO_JSON: Caminho do arquivo estruturado legado/base de prontuários.
- ARQUIVO_DINAMICO: Caminho do JSON de persistência dinâmica de novos registros clínicos.
- DIRETORIO_CHROMA: Caminho no disco para persistência da base vetorial ChromaDB.
"""
import os
from dotenv import load_dotenv

# Carrega variáveis definidas no arquivo .env local, se existir
load_dotenv()

# --- MODELOS E INTELIGÊNCIA ARTIFICIAL ---
# Modelo multilíngue para geração de representações semânticas densas (embeddings)
MODELO_EMBEDDING = os.getenv("MODELO_EMBEDDING", "paraphrase-multilingual-MiniLM-L12-v2")
DIMENSAO_EMBEDDING = 384

# Nome do modelo LLM disponível na instância do Ollama
# No ambiente remoto de collocation (192.168.88.2), utilizamos 'llama3:latest'
MODELO_LLM = os.getenv("MODELO_LLM", "llama3:latest")

# Endpoint REST da API do Ollama
# No ambiente com VPN (tun0), aponte para http://192.168.88.2:11434/api/generate
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Quantidade de camadas da rede neural descarregadas na VRAM da GPU (0 = 100% CPU)
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "22"))

# Janela de contexto em tokens enviada ao Ollama (Llama 3 suporta até 8192; 4096 é o padrão recomendado)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# --- MENSAGERIA E ARQUITETURA ASSÍNCRONA ---
# URL de conexão AMQP para o message broker RabbitMQ (padrão RPC)
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")

# Limite de mensagens recentes armazenadas no histórico de diálogo de cada paciente
LIMITE_HISTORICO = int(os.getenv("LIMITE_HISTORICO", "10"))

# Quantidade padrão de chunks recuperados pelo ChromaDB por consulta de similaridade
TOP_K_BUSCA = int(os.getenv("TOP_K_BUSCA", "16"))

# --- DIRETÓRIOS E PERSISTÊNCIA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Arquivo base de dados de prontuários
ARQUIVO_JSON = os.getenv("ARQUIVO_JSON", os.path.join(os.path.dirname(BASE_DIR), "prontuario_estruturado.json"))

# Arquivo JSON que acumula registros adicionados dinamicamente via POST /api/prontuario
ARQUIVO_DINAMICO = os.getenv("ARQUIVO_DINAMICO", os.path.join(BASE_DIR, "prontuarios_dinamicos.json"))

# Diretório de persistência do ChromaDB (banco vetorial SQLite/HNSW)
DIRETORIO_CHROMA = os.getenv("DIRETORIO_CHROMA", os.path.join(os.path.dirname(BASE_DIR), "chroma_db"))
