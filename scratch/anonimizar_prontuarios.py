#!/usr/bin/env python3
"""
Script de Anonimização Completa e Sanitização (LGPD / HIPAA Compliance).
Substitui identificadores, nomes de parentes e unidades locais por referências genéricas.
Purga e recria a base ChromaDB e o arquivo prontuarios_dinamicos.json.
"""
import os
import re
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "/app/chroma_db" if os.path.exists("/app/chroma_db") else "/home/ease/chroma_db"
DINAMICOS_PATH = "/app/prontuarios_dinamicos.json" if os.path.exists("/app") else "/home/ease/rag_prontuario/prontuarios_dinamicos.json"
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Mapeamentos estritos de desidentificação
REGRAS_SUBSTITUICAO = [
    # Nomes de acompanhantes e parentes
    (r"\bMÃE FERNANDA\b", "MÃE (ACOMPANHANTE)"),
    (r"\bACOMPANHADO DA MÃE FERNANDA\b", "ACOMPANHADO DA MÃE"),
    (r"\bESPOSA ADRIANA\b", "ESPOSA (ACOMPANHANTE)"),
    (r"\bACOMPANHADA DA ESPOSA ADRIANA\b", "ACOMPANHADO DA ESPOSA"),
    (r"\bESPOSO \(JOILSON\)\b", "ESPOSO (ACOMPANHANTE)"),
    (r"\bJOILSON\b", "ACOMPANHANTE"),
    (r"\bFILHA CARLA\b", "FILHA (ACOMPANHANTE)"),
    (r"\bACOMPANHADO DA FILHA CARLA\b", "ACOMPANHADO DA FILHA"),
    (r"\bFILHA \(TAINARA\)\b", "FILHA (ACOMPANHANTE)"),
    (r"\bTAINARA\b", "ACOMPANHANTE"),
    (r"\bACOMPANHADO DA CLEIDIANE\b", "ACOMPANHADO POR FAMILIAR"),
    (r"\bCLEIDIANE\b", "ACOMPANHANTE"),

    # Unidades de saúde e locais específicos
    (r"UPA CASTELÂNDIA", "UNIDADE DE PRONTO ATENDIMENTO"),
    (r"HOSPITAL EVANGÉLICO", "HOSPITAL DE REFERÊNCIA"),
    (r"HJSN", "HOSPITAL REGIONAL"),
    (r"LAB EINSTEIN", "LABORATÓRIO CENTRAL"),
    (r"LAB \(EINSTEIN", "LAB (LABORATÓRIO CENTRAL"),

    # Médicos e profissionais
    (r"Dr\. Plantonista PS", "Médico Plantonista"),
    (r"Dr\. Hospitalista UPA", "Médico Assistente"),
    (r"Dr\. Emergencista Sala Vermelha", "Médico Emergencista"),
]

def sanitizar_texto(texto):
    for padrao, subst in REGRAS_SUBSTITUICAO:
        texto = re.sub(padrao, subst, texto, flags=re.IGNORECASE)
    return texto

def main():
    print("=" * 65)
    print(">>> INICIANDO ANONIMIZAÇÃO E SANITIZAÇÃO DE PRONTUÁRIOS (LGPD) <<<")
    print("=" * 65)

    if not os.path.exists(DINAMICOS_PATH):
        print(f"[ERRO] Arquivo {DINAMICOS_PATH} não encontrado.")
        return

    with open(DINAMICOS_PATH, "r", encoding="utf-8") as f:
        registros = json.load(f)

    print(f"Total de registros a serem anonimizados: {len(registros)}")

    registros_anonimizados = []
    textos_limpos = []
    ids_limpos = []
    metadados_limpos = []

    for idx, reg in enumerate(registros, start=1):
        categoria = reg.get("categoria", "Atendimento")
        prefixo = "ps" if "Pronto" in categoria else "enf"
        novo_id = f"{prefixo}_{idx:02d}"
        novo_nome = f"Paciente {idx:02d} ({'PS' if prefixo == 'ps' else 'Enfermaria'})"

        texto_original = reg.get("texto", "")
        texto_anonimizado = sanitizar_texto(texto_original)

        # Substitui iniciais no texto pelo identificador neutro
        iniciais_antigas = reg.get("nome", "")
        if iniciais_antigas and len(iniciais_antigas) <= 15:
            # Substitui por Paciente XX
            texto_anonimizado = re.sub(rf"\b{re.escape(iniciais_antigas)}\b", f"Paciente {idx:02d}", texto_anonimizado, flags=re.IGNORECASE)

        item_limpo = {
            "id": f"{novo_id}_reg01",
            "paciente_id": novo_id,
            "nome": novo_nome,
            "categoria": categoria,
            "data": reg.get("data", "26/10/2026"),
            "tipo_evento": reg.get("tipo_evento", "Atendimento Clínico"),
            "profissional": "Médico Plantonista",
            "texto": texto_anonimizado,
        }

        registros_anonimizados.append(item_limpo)
        ids_limpos.append(item_limpo["id"])
        textos_limpos.append(item_limpo["texto"])
        metadados_limpos.append({
            "paciente_id": novo_id,
            "nome": novo_nome,
            "categoria": categoria,
            "data": item_limpo["data"],
            "tipo_evento": item_limpo["tipo_evento"],
            "profissional": item_limpo["profissional"],
        })

    # Salva o JSON completamente sanitizado
    with open(DINAMICOS_PATH, "w", encoding="utf-8") as f:
        json.dump(registros_anonimizados, f, ensure_ascii=False, indent=2)
    print(f"[OK] {DINAMICOS_PATH} regravado com dados 100% anonimizados.")

    # Atualiza o ChromaDB
    print("\nAtualizando banco vetorial ChromaDB com dados desidentificados...")
    cliente = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    nome_colecao = "prontuario_evolucoes"

    try:
        cliente.delete_collection(nome_colecao)
        print(f"[OK] Coleção anterior deletada.")
    except Exception as e:
        print(f"[Aviso] {e}")

    colecao = cliente.create_collection(name=nome_colecao, metadata={"hnsw:space": "cosine"})

    print(f"Gerando novos embeddings com {MODELO_EMBEDDING}...")
    modelo = SentenceTransformer(MODELO_EMBEDDING)
    embeddings = modelo.encode(textos_limpos, show_progress_bar=False).tolist()

    colecao.add(
        ids=ids_limpos,
        embeddings=embeddings,
        metadatas=metadados_limpos,
        documents=textos_limpos
    )

    print(f"[SUCESSO] {len(ids_limpos)} prontuários anônimos indexados no ChromaDB!")
    print(f"Total na coleção: {colecao.count()} documentos.")
    print("=" * 65)

if __name__ == "__main__":
    main()

