#!/usr/bin/env python3
"""
Bateria de testes clínicos com pacientes reais (PS vs Enfermaria UPA).
Mede a precisão da recuperação vetorial (RAG) e a geração de respostas/resumo.
"""
import time
import requests

OLLAMA_LOCAL_URL = "http://127.0.0.1:11434/api/generate"
MODELO = "llama3:latest"

CASOS_TESTE = [
    {
        "paciente_id": "ps_fs",
        "nome": "F.S. (Pronto Atendimento)",
        "pergunta": "O paciente F.S. tem alguma alergia medicamentosa? E qual foi a queixa principal?",
        "chaves_esperadas": ["Bromoprida", "cefaleia", "dor abdominal", "epigastro"],
    },
    {
        "paciente_id": "ps_fs",
        "nome": "F.S. (Pronto Atendimento - Exames)",
        "pergunta": "Quais exames laboratoriais vieram alterados em F.S. e qual a conduta?",
        "chaves_esperadas": ["TGP", "FA", "111", "457", "sintomáticos"],
    },
    {
        "paciente_id": "ps_tp",
        "nome": "T.P. (Pronto Atendimento - Alergia Grave)",
        "pergunta": "Quais remédios o paciente T.P. NÃO pode tomar de jeito nenhum?",
        "chaves_esperadas": ["AAS", "Dipirona", "AINEs", "Fenilbutazona"],
    },
    {
        "paciente_id": "upa_ag",
        "nome": "A.G. (Admissão Enfermaria UPA)",
        "pergunta": "Qual o motivo da internação de A.G. e qual comorbidade prévia ele trata?",
        "chaves_esperadas": ["Hepatite medicamentosa", "ITU", "Hanseníase"],
    },
    {
        "paciente_id": "upa_ag",
        "nome": "A.G. (Bilirrubinas e Conduta)",
        "pergunta": "Como estão as bilirrubinas de A.G. e qual antibiótico foi prescrito na internação?",
        "chaves_esperadas": ["9,90", "Ceftriaxona"],
    },
    {
        "paciente_id": "upa_jg",
        "nome": "J.G. (Sala Vermelha UPA)",
        "pergunta": "Qual medicação em bomba de infusão foi iniciada para J.G. e qual a evolução do quadro?",
        "chaves_esperadas": ["Nipride", "133/86", "desmame"],
    },
    {
        "paciente_id": "upa_av",
        "nome": "A.V. (Pancreatite Medicamentosa)",
        "pergunta": "Qual produto a paciente A.V. estava usando que causou a suspeita de pancreatite?",
        "chaves_esperadas": ["Mounjaro Natural", "Amilase", "Lipase"],
    }
]

def buscar_contexto_api(paciente_id, pergunta):
    url = f"http://localhost:8000/api/prontuario/{paciente_id}"
    r = requests.get(url, timeout=5)
    dados = r.json()
    registros = dados.get("registros", [])
    if not registros:
        return ""
    return "\n\n".join([f"[{reg.get('data', '')}] {reg.get('texto', '')}" for reg in registros])

def consultar_llm(contexto, pergunta):
    prompt = f"""Você é um médico assistente clínico experiente.
Responda de forma direta, técnica, precisa e objetiva à pergunta do médico com base EXCLUSIVAMENTE nas informações do prontuário abaixo.

PRONTUÁRIO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA CLÍNICA:"""

    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 256,
        }
    }
    t0 = time.time()
    res = requests.post(OLLAMA_LOCAL_URL, json=payload, timeout=60)
    duracao = time.time() - t0
    resposta = res.json().get("response", "").strip()
    return resposta, duracao

def gerar_resumo_llm(paciente_id):
    contexto = buscar_contexto_api(paciente_id, "")
    prompt = f"""Você é um médico assistente clínico.
Gere um RESUMO CLÍNICO ESTRUTURADO para o prontuário abaixo, contendo:
1. Identificação e Motivo do Atendimento / Queixa Principal
2. História Clínica, Comorbidades e Alergias
3. Achados de Exame Físico e Exames Complementares (com valores relevantes)
4. Diagnóstico (HD) e Conduta Médica (CD)

PRONTUÁRIO:
{contexto}

RESUMO CLÍNICO:"""

    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512,
        }
    }
    t0 = time.time()
    res = requests.post(OLLAMA_LOCAL_URL, json=payload, timeout=90)
    duracao = time.time() - t0
    return res.json().get("response", "").strip(), duracao

def main():
    print("=" * 70)
    print(">>> BATERIA CLÍNICA: RAG COM PRONTUÁRIOS REAIS DE PS E UPA <<<")
    print(f"Modelo LLM: {MODELO} (Local: {OLLAMA_LOCAL_URL})")
    print("=" * 70)

    acertos = 0
    total = len(CASOS_TESTE)

    for i, caso in enumerate(CASOS_TESTE, start=1):
        print(f"\n[{i}/{total}] Testando Paciente: {caso['nome']} ({caso['paciente_id']})")
        print(f"  Pergunta: {caso['pergunta']}")

        contexto = buscar_contexto_api(caso['paciente_id'], caso['pergunta'])
        if not contexto:
            print("  ❌ ERRO: Contexto não encontrado na API!")
            continue

        resp, tempo = consultar_llm(contexto, caso['pergunta'])
        print(f"  ⏱ Tempo de inferência: {tempo:.2f}s")
        print(f"  Resposta LLM: {resp[:180]}...")

        # Validação das palavras-chave
        encontrados = [k for k in caso['chaves_esperadas'] if k.lower() in resp.lower()]
        taxa = len(encontrados) / len(caso['chaves_esperadas'])
        print(f"  Precisão semântica: {len(encontrados)}/{len(caso['chaves_esperadas'])} termos encontrados ({encontrados})")

        if taxa >= 0.5:
            print("  ✅ SUCESSO")
            acertos += 1
        else:
            print("  ⚠️ PARCIAL OU INCOMPLETO")

    print("\n" + "=" * 70)
    print(">>> TESTE DE GERAÇÃO DE RESUMO CLÍNICO ESTRUTURADO <<<")
    print("=" * 70)
    for pac_id in ["ps_fs", "upa_ag", "upa_jg"]:
        print(f"\n--- Gerando Resumo para: {pac_id} ---")
        resumo, t_res = gerar_resumo_llm(pac_id)
        print(f"⏱ Tempo para resumir: {t_res:.2f} segundos")
        print(f"Resumo Gerado:\n{resumo}")

    print("\n" + "=" * 70)
    print(f"RESULTADO FINAL: {acertos}/{total} perguntas respondidas com sucesso clínico!")
    print("=" * 70)

if __name__ == "__main__":
    main()

