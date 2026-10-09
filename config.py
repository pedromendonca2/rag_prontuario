import os

MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
DIMENSAO_EMBEDDING = 384
MODELO_LLM = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
LIMITE_HISTORICO = 10
TOP_K_BUSCA = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_JSON = os.path.join(os.path.dirname(BASE_DIR), "prontuario_estruturado.json")
DIRETORIO_CHROMA = os.path.join(os.path.dirname(BASE_DIR), "chroma_db")
