import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

import config


class IndexadorVectorial:
    def __init__(self, diretorio_chroma=None, modelo_embedding=None):
        self.diretorio_chroma = diretorio_chroma or config.DIRETORIO_CHROMA
        self.modelo_nome = modelo_embedding or config.MODELO_EMBEDDING
        self.dimensao = config.DIMENSAO_EMBEDDING

        print(f"[indexador] Carregando modelo de embedding: {self.modelo_nome}")
        self.modelo = SentenceTransformer(self.modelo_nome)

        os.makedirs(self.diretorio_chroma, exist_ok=True)
        self.cliente = chromadb.PersistentClient(
            path=self.diretorio_chroma,
            settings=Settings(anonymized_telemetry=False),
        )

        nome_colecao = "prontuario_evolucoes"
        colecoes_existentes = [c.name for c in self.cliente.list_collections()]
        if nome_colecao in colecoes_existentes:
            self.colecao = self.cliente.get_collection(nome_colecao)
            print(f"[indexador] Colecao '{nome_colecao}' carregada ({self.colecao.count()} documentos)")
        else:
            self.colecao = self.cliente.create_collection(
                name=nome_colecao,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[indexador] Colecao '{nome_colecao}' criada")

    def _gerar_embeddings(self, textos):
        return self.modelo.encode(textos, show_progress_bar=True).tolist()

    def indexar(self, registros):
        textos = [r["texto_bruto"] for r in registros]
        ids = [str(r["id"]) for r in registros]
        metadados = [{"data": r["data"], "tipo_evento": r["tipo_evento"]} for r in registros]

        embeddings = self._gerar_embeddings(textos)

        self.colecao.delete(where={"tipo_evento": {"$ne": ""}})

        batch_size = 32
        for i in range(0, len(ids), batch_size):
            fim = min(i + batch_size, len(ids))
            self.colecao.add(
                ids=ids[i:fim],
                embeddings=embeddings[i:fim],
                metadatas=metadados[i:fim],
                documents=textos[i:fim],
            )

        print(f"[indexador] Indexados {len(ids)} documentos no ChromaDB")

    def buscar(self, pergunta, k=None):
        if k is None:
            k = config.TOP_K_BUSCA
        embedding_pergunta = self.modelo.encode([pergunta]).tolist()
        resultados = self.colecao.query(
            query_embeddings=embedding_pergunta,
            n_results=min(k, self.colecao.count()),
        )
        return resultados

    def buscar_por_texto(self, pergunta, k=None):
        resultados = self.buscar(pergunta, k)
        documentos = resultados["documents"][0] if resultados["documents"] else []
        metadados = resultados["metadatas"][0] if resultados["metadatas"] else []
        distancias = resultados["distances"][0] if resultados["distances"] else []
        return list(zip(documentos, metadados, distancias))

    def contar(self):
        return self.colecao.count()
