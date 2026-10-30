"""
Módulo de Indexação e Busca Vetorial com ChromaDB e SentenceTransformers.

Responsável por:
1. Carregar o modelo local de embeddings semânticos densos (SentenceTransformers).
2. Gerenciar a coleção 'prontuario_evolucoes' no banco vetorial ChromaDB com distância cosseno.
3. Garantir isolamento rigoroso por paciente através de metadados ('paciente_id').
4. Executar buscas semânticas simples e multi-queries para recuperar o contexto clínico relevante.
5. Permitir a ingestão dinâmica em tempo de execução de novas evoluções médicas.
"""
import os
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

import config


class IndexadorVectorial:
    """
    Gerenciador da coleção vetorial de prontuários médicos no ChromaDB.
    """

    def __init__(self, diretorio_chroma=None, modelo_embedding=None):
        """
        Inicializa o modelo de embeddings e estabelece conexão com o ChromaDB.

        Args:
            diretorio_chroma (str, opcional): Caminho para a pasta onde o SQLite/HNSW do Chroma é persistido.
            modelo_embedding (str, opcional): Nome do modelo no HuggingFace.
        """
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
            print(f"[indexador] Coleção '{nome_colecao}' carregada com sucesso ({self.colecao.count()} documentos)")
        else:
            # Configuração com métrica cosseno para melhor representação semântica de textos médicos
            self.colecao = self.cliente.create_collection(
                name=nome_colecao,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[indexador] Nova coleção '{nome_colecao}' criada com métrica cosseno")

    def _gerar_embeddings(self, textos: list) -> list:
        """
        Calcula os vetores de embedding para uma lista de strings.

        Args:
            textos (list[str]): Lista de trechos ou evoluções médicas.

        Returns:
            list[list[float]]: Lista de vetores densos (dimensão 384).
        """
        return self.modelo.encode(textos, show_progress_bar=False).tolist()

    def indexar(self, registros: list):
        """
        Indexa em lote uma coleção inicial de registros médicos no ChromaDB.

        Atenção: Limpa os registros preexistentes antes de reinserir em lotes (batching).

        Args:
            registros (list[dict]): Lista de dicionários contendo ao menos:
                - 'texto_bruto': Conteúdo textual da evolução.
                - 'id': Identificador único do registro.
                - 'data': Data no formato DD/MM/AAAA.
                - 'tipo_evento': Tipo do registro (ex: Consulta, PS).
                - 'paciente_id': Matrícula de isolamento do paciente.
        """
        textos = [r["texto_bruto"] for r in registros]
        ids = [str(r["id"]) for r in registros]
        metadados = [
            {
                "data": r.get("data", ""),
                "tipo_evento": r.get("tipo_evento", ""),
                "paciente_id": str(r.get("paciente_id", "default_patient")),
            }
            for r in registros
        ]

        embeddings = self._gerar_embeddings(textos)

        # Remove dados antigos para garantir reindexação limpa
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

        print(f"[indexador] Indexados {len(ids)} documentos no ChromaDB em lotes de {batch_size}")

    def adicionar_registro(self, texto: str, paciente_id: str, data=None, tipo_evento=None, profissional=None, id_registro=None) -> dict:
        """
        Insere dinamicamente uma nova evolução médica no banco vetorial com isolamento de paciente.

        Gera o embedding imediatamente e grava o documento e seus metadados no ChromaDB.

        Args:
            texto (str): Conteúdo textual da evolução médica.
            paciente_id (str): Matrícula ou ID do paciente (filtro de isolamento obrigatório).
            data (str, opcional): Data no formato "DD/MM/AAAA HH:MM" ou "DD/MM/AAAA". Se omitida, usa agora.
            tipo_evento (str, opcional): Categoria do evento (padrão: "Evolução Médica").
            profissional (str, opcional): Nome ou CRM do profissional de saúde autor da nota.
            id_registro (str, opcional): ID exclusivo; se omitido, gera prefixo com paciente_id e UUID.

        Returns:
            dict: Objeto estruturado com os dados salvos do registro.
        """
        if not paciente_id or not str(paciente_id).strip():
            raise ValueError("O campo 'paciente_id' (matrícula) é obrigatório.")
        if not texto or not str(texto).strip():
            raise ValueError("O conteúdo da evolução ('texto') é obrigatório.")

        paciente_id_str = str(paciente_id).strip()
        texto_limpo = str(texto).strip()

        if not data:
            data = datetime.now().strftime("%d/%m/%Y %H:%M")
        if not tipo_evento:
            tipo_evento = "Evolução Médica"
        if not id_registro:
            id_registro = f"{paciente_id_str}_{uuid.uuid4().hex[:8]}"

        metadata = {
            "paciente_id": paciente_id_str,
            "data": str(data),
            "tipo_evento": str(tipo_evento),
            "profissional": str(profissional or ""),
        }

        embeddings = self._gerar_embeddings([texto_limpo])

        self.colecao.add(
            ids=[id_registro],
            embeddings=embeddings,
            metadatas=[metadata],
            documents=[texto_limpo],
        )

        print(f"[indexador] Novo registro adicionado (ID: {id_registro}, Paciente: {paciente_id_str}, Data: {data})")
        return {
            "id": id_registro,
            "paciente_id": paciente_id_str,
            "data": data,
            "tipo_evento": tipo_evento,
            "profissional": profissional or "",
            "texto": texto_limpo,
        }

    def buscar(self, pergunta: str, k=None, where=None) -> dict:
        """
        Executa a busca vetorial bruta no ChromaDB para uma pergunta em linguagem natural.

        Args:
            pergunta (str): Termo de consulta ou pergunta clínica.
            k (int, opcional): Quantidade de chunks mais similares a recuperar.
            where (dict, opcional): Filtro por metadados (ex: {"paciente_id": "123"}).

        Returns:
            dict: Estrutura padrão do ChromaDB contendo 'documents', 'metadatas' e 'distances'.
        """
        if k is None:
            k = config.TOP_K_BUSCA

        if where:
            existentes = self.colecao.get(where=where)
            total = len(existentes.get("ids", []))
            if total == 0:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            n_results = min(k, total)
        else:
            total = self.colecao.count()
            if total == 0:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            n_results = min(k, total)

        embedding_pergunta = self.modelo.encode([pergunta], show_progress_bar=False).tolist()
        resultados = self.colecao.query(
            query_embeddings=embedding_pergunta,
            n_results=n_results,
            where=where,
        )
        return resultados

    def buscar_por_texto(self, pergunta: str, k=None, where=None) -> list:
        """
        Executa busca vetorial e retorna uma lista limpa de tuplas (texto, metadados, distancia).

        Args:
            pergunta (str): Texto da consulta.
            k (int, opcional): Limite de resultados.
            where (dict, opcional): Filtro de metadados (ex: por paciente).

        Returns:
            list[tuple[str, dict, float]]: Lista de tuplas com os dados recuperados.
        """
        resultados = self.buscar(pergunta, k, where=where)
        documentos = resultados["documents"][0] if resultados.get("documents") else []
        metadados = resultados["metadatas"][0] if resultados.get("metadatas") else []
        distancias = resultados["distances"][0] if resultados.get("distances") else []
        return list(zip(documentos, metadados, distancias))

    def buscar_multiplas_queries(self, perguntas: list, k=None, where=None) -> list:
        """
        Gera embeddings e consulta o ChromaDB para múltiplas sub-queries simultâneas.

        Muito útil quando a pergunta do médico é composta por múltiplos questionamentos
        (ex: 'Qual a PA na admissão? Quais remédios toma? Teve alta?').

        Args:
            perguntas (list[str]): Lista de sub-perguntas a serem buscadas.
            k (int, opcional): Quantidade de resultados por sub-query.
            where (dict, opcional): Filtro por metadados de isolamento do paciente.

        Returns:
            list[tuple[str, dict, float]]: Lista unificada de tuplas (documento, metadados, distância).
        """
        if not perguntas:
            return []
        if k is None:
            k = config.TOP_K_BUSCA

        if where:
            existentes = self.colecao.get(where=where)
            total = len(existentes.get("ids", []))
            if total == 0:
                return []
            n_results = min(k, total)
        else:
            total = self.colecao.count()
            if total == 0:
                return []
            n_results = min(k, total)

        embeddings = self.modelo.encode(perguntas, show_progress_bar=False).tolist()
        resultados = self.colecao.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where=where,
        )

        todos_itens = []
        docs_list = resultados.get("documents", [])
        metas_list = resultados.get("metadatas", [])
        dists_list = resultados.get("distances", [])

        for q_idx in range(len(perguntas)):
            docs = docs_list[q_idx] if q_idx < len(docs_list) else []
            metas = metas_list[q_idx] if q_idx < len(metas_list) else []
            dists = dists_list[q_idx] if q_idx < len(dists_list) else []
            for d, m, dist in zip(docs, metas, dists):
                todos_itens.append((d, m, dist))

        return todos_itens

    def listar_registros_paciente(self, paciente_id: str) -> list:
        """
        Recupera e ordena cronologicamente todas as evoluções médicas de um paciente.

        Utilizado pela interface web (Linha do Tempo) e para montagem do contexto integral
        no momento de gerar o Resumo Clínico Estruturado.

        Args:
            paciente_id (str): Matrícula ou ID único do paciente.

        Returns:
            list[dict]: Lista de evoluções ordenadas da mais antiga para a mais recente.
        """
        if not paciente_id:
            return []
        dados = self.colecao.get(where={"paciente_id": str(paciente_id).strip()})
        if not dados or not dados.get("documents"):
            return []

        itens = []
        for id_, doc, meta in zip(dados["ids"], dados["documents"], dados["metadatas"]):
            itens.append({
                "id": id_,
                "texto": doc,
                "data": meta.get("data", ""),
                "tipo_evento": meta.get("tipo_evento", ""),
                "profissional": meta.get("profissional", ""),
                "paciente_id": meta.get("paciente_id", str(paciente_id)),
            })

        def parse_data(item):
            d_str = item.get("data", "")
            try:
                # Tenta parsear formato "DD/MM/AAAA HH:MM" ou "DD/MM/AAAA"
                d_parte = d_str.split()[0]
                return datetime.strptime(d_parte, "%d/%m/%Y")
            except Exception:
                return datetime.min

        # Ordena cronologicamente para exibição fidedigna da história do paciente
        itens.sort(key=parse_data)
        return itens

    def contar(self) -> int:
        """Retorna o número total de documentos armazenados na coleção."""
        return self.colecao.count()
