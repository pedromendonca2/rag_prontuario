import time
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_matricula_validation():
    print(">>> [TESTE 1] Validação de Matrícula Obrigatória")

    # 1.1 POST /api/prontuario sem paciente_id
    r1 = requests.post(f"{BASE_URL}/api/prontuario", json={"texto": "Consulta teste"})
    assert r1.status_code == 422 or r1.status_code == 400, f"Falhou: esperado 400/422, obteve {r1.status_code}"
    print("  ✓ POST /api/prontuario sem paciente_id rejeitado.")

    # 1.2 POST /api/prontuario com paciente_id vazio
    r2 = requests.post(f"{BASE_URL}/api/prontuario", json={"paciente_id": "   ", "texto": "Consulta teste"})
    assert r2.status_code == 400, f"Falhou: esperado 400, obteve {r2.status_code}"
    print("  ✓ POST /api/prontuario com paciente_id em branco rejeitado (400).")

    # 1.3 POST /api/prontuario com texto vazio
    r3 = requests.post(f"{BASE_URL}/api/prontuario", json={"paciente_id": "mat_100", "texto": "   "})
    assert r3.status_code == 400, f"Falhou: esperado 400, obteve {r3.status_code}"
    print("  ✓ POST /api/prontuario com texto vazio rejeitado (400).")

    # 1.4 POST /api/chat sem session_id
    r4 = requests.post(f"{BASE_URL}/api/chat", json={"pergunta": "Olá"})
    assert r4.status_code == 422 or r4.status_code == 400, f"Falhou: esperado 400/422, obteve {r4.status_code}"
    print("  ✓ POST /api/chat sem session_id rejeitado.")

    # 1.5 POST /api/resumo sem session_id
    r5 = requests.post(f"{BASE_URL}/api/resumo", json={})
    assert r5.status_code == 422 or r5.status_code == 400, f"Falhou: esperado 400/422, obteve {r5.status_code}"
    print("  ✓ POST /api/resumo sem session_id rejeitado.")

def test_add_records_and_listing():
    print("\n>>> [TESTE 2] Adição de Evoluções e Listagem Cronológica")

    paciente_a = "mat_5001"
    paciente_b = "mat_6002"

    # Paciente A (Maria Oliveira)
    doc_a1 = {
        "paciente_id": paciente_a,
        "texto": "10/01/2026 - Consulta: Paciente Maria Oliveira, 52 anos. Diagnóstico de Diabetes Mellitus tipo 2 e Dislipidemia. Prescrito Metformina 850mg 2x ao dia e Atorvastatina 20mg ao dia. Glicemia em jejum 165 mg/dL e HbA1c 7.8%. PA 130x80 mmHg.",
        "data": "10/01/2026",
        "tipo_evento": "Consulta",
        "profissional": "Dra. Carolina Lima (CRM 12345)"
    }
    r_a1 = requests.post(f"{BASE_URL}/api/prontuario", json=doc_a1)
    assert r_a1.status_code == 201, f"Falha ao criar registro A1: {r_a1.text}"
    print(f"  ✓ Registrada consulta 1 para Maria ({paciente_a}).")

    doc_a2 = {
        "paciente_id": paciente_a,
        "texto": "<p>15/02/2026 - Retorno: Paciente Maria Oliveira relata <strong>excelente adaptação</strong> à Metformina. Nega efeitos colaterais. Glicemia matinal controlada em 110 mg/dL. PA 124x76 mmHg.<br/>Mantida conduta.</p>",
        "data": "15/02/2026",
        "tipo_evento": "Retorno",
        "profissional": "Dra. Carolina Lima (CRM 12345)"
    }
    r_a2 = requests.post(f"{BASE_URL}/api/prontuario", json=doc_a2)
    assert r_a2.status_code == 201, f"Falha ao criar registro A2: {r_a2.text}"
    print(f"  ✓ Registrado retorno 2 para Maria ({paciente_a}) com sanitização HTML.")

    # Paciente B (Carlos Souza)
    doc_b1 = {
        "paciente_id": paciente_b,
        "texto": "05/03/2026 - Consulta: Paciente Carlos Souza, 34 anos. Queixa de dispneia e sibilos aos esforços. Diagnóstico clínico de Asma Brônquica moderada. Prescrito Budesonida 200mcg inalatória 12/12h e Salbutamol 100mcg spray para resgate.",
        "data": "05/03/2026",
        "tipo_evento": "Consulta",
        "profissional": "Dr. Fernando Mendes (CRM 67890)"
    }
    r_b1 = requests.post(f"{BASE_URL}/api/prontuario", json=doc_b1)
    assert r_b1.status_code == 201, f"Falha ao criar registro B1: {r_b1.text}"
    print(f"  ✓ Registrada consulta 1 para Carlos ({paciente_b}).")

    # Listagem Paciente A
    r_list_a = requests.get(f"{BASE_URL}/api/prontuario/{paciente_a}")
    assert r_list_a.status_code == 200
    dados_a = r_list_a.json()
    assert dados_a["total"] >= 2, f"Esperado >= 2 registros para A, obteve {dados_a['total']}"
    print(f"  ✓ Listagem de Maria confirmada: {dados_a['total']} registros.")

    # Listagem Paciente B
    r_list_b = requests.get(f"{BASE_URL}/api/prontuario/{paciente_b}")
    assert r_list_b.status_code == 200
    dados_b = r_list_b.json()
    assert dados_b["total"] >= 1, f"Esperado >= 1 registros para B, obteve {dados_b['total']}"
    print(f"  ✓ Listagem de Carlos confirmada: {dados_b['total']} registros.")

    # Listagem Paciente Inexistente
    r_list_none = requests.get(f"{BASE_URL}/api/prontuario/mat_fantasma_9999")
    assert r_list_none.status_code == 200
    dados_none = r_list_none.json()
    assert dados_none["total"] == 0
    print("  ✓ Listagem de paciente inexistente retornou 0 registros.")

def test_chat_isolation():
    print("\n>>> [TESTE 3] Isolamento RAG no Chat entre Pacientes")

    paciente_a = "mat_5001"
    paciente_b = "mat_6002"
    paciente_default = "default_patient"
    paciente_vazio = "mat_sem_historico_7777"

    # Pergunta para Maria
    t0 = time.time()
    r_chat_a = requests.post(f"{BASE_URL}/api/chat", json={
        "session_id": paciente_a,
        "pergunta": "Qual o diagnóstico e quais remédios a paciente toma?"
    })
    elapsed_a = time.time() - t0
    assert r_chat_a.status_code == 200, f"Erro chat A: {r_chat_a.text}"
    ans_a = r_chat_a.json()["resposta"]
    print(f"  [Maria - {elapsed_a:.2f}s]: {ans_a}")
    assert "metformina" in ans_a.lower() or "diabetes" in ans_a.lower(), "Deveria citar Metformina/Diabetes"
    assert "salbutamol" not in ans_a.lower() and "losartana" not in ans_a.lower(), "VAZAMENTO DE DADOS: citou remédio de outro paciente!"
    print("  ✓ Chat de Maria validado sem vazamento.")

    # Pergunta para Carlos
    t0 = time.time()
    r_chat_b = requests.post(f"{BASE_URL}/api/chat", json={
        "session_id": paciente_b,
        "pergunta": "Qual a queixa principal e qual remédio de resgate foi prescrito?"
    })
    elapsed_b = time.time() - t0
    assert r_chat_b.status_code == 200, f"Erro chat B: {r_chat_b.text}"
    ans_b = r_chat_b.json()["resposta"]
    print(f"  [Carlos - {elapsed_b:.2f}s]: {ans_b}")
    assert "salbutamol" in ans_b.lower() or "asma" in ans_b.lower(), "Deveria citar Salbutamol/Asma"
    assert "metformina" not in ans_b.lower() and "losartana" not in ans_b.lower(), "VAZAMENTO DE DADOS: citou remédio de outro paciente!"
    print("  ✓ Chat de Carlos validado sem vazamento.")

    # Pergunta para João dos Santos (default_patient)
    t0 = time.time()
    r_chat_def = requests.post(f"{BASE_URL}/api/chat", json={
        "session_id": paciente_default,
        "pergunta": "Qual a medicação ativa do paciente para pressão arterial?"
    })
    elapsed_def = time.time() - t0
    assert r_chat_def.status_code == 200, f"Erro chat default: {r_chat_def.text}"
    ans_def = r_chat_def.json()["resposta"]
    print(f"  [João dos Santos - {elapsed_def:.2f}s]: {ans_def}")
    assert "losartana" in ans_def.lower(), "Deveria citar Losartana"
    assert "metformina" not in ans_def.lower() and "salbutamol" not in ans_def.lower(), "VAZAMENTO DE DADOS!"
    print("  ✓ Chat de João dos Santos validado.")

    # Pergunta para Paciente sem histórico
    r_chat_vazio = requests.post(f"{BASE_URL}/api/chat", json={
        "session_id": paciente_vazio,
        "pergunta": "Como está a pressão dele?"
    })
    assert r_chat_vazio.status_code == 200
    ans_vazio = r_chat_vazio.json()["resposta"]
    print(f"  [Sem histórico]: {ans_vazio}")
    assert "não há registros" in ans_vazio.lower() or "não encontrei" in ans_vazio.lower(), "Deveria avisar que não há registros"
    print("  ✓ Chat de paciente vazio avisou corretamente ausência de registros.")

def test_resumo_isolation():
    print("\n>>> [TESTE 4] Isolamento no Resumo Clínico")

    paciente_a = "mat_5001"
    paciente_b = "mat_6002"
    paciente_vazio = "mat_sem_historico_7777"

    # Resumo Paciente A
    t0 = time.time()
    r_resumo_a = requests.post(f"{BASE_URL}/api/resumo", json={"session_id": paciente_a})
    elapsed_a = time.time() - t0
    assert r_resumo_a.status_code == 200, f"Erro resumo A: {r_resumo_a.text}"
    res_a = r_resumo_a.json()["resumo"]
    print(f"  [Resumo Maria - {elapsed_a:.2f}s]:\n{res_a[:250]}...")
    assert "metformina" in res_a.lower() or "diabetes" in res_a.lower(), "Resumo de Maria deveria conter Diabetes/Metformina"
    assert "salbutamol" not in res_a.lower(), "Resumo de Maria não pode ter Salbutamol"

    # Resumo Paciente B
    t0 = time.time()
    r_resumo_b = requests.post(f"{BASE_URL}/api/resumo", json={"session_id": paciente_b})
    elapsed_b = time.time() - t0
    assert r_resumo_b.status_code == 200, f"Erro resumo B: {r_resumo_b.text}"
    res_b = r_resumo_b.json()["resumo"]
    print(f"  [Resumo Carlos - {elapsed_b:.2f}s]:\n{res_b[:250]}...")
    assert "asma" in res_b.lower() or "budesonida" in res_b.lower() or "salbutamol" in res_b.lower(), "Resumo de Carlos deveria conter Asma"
    assert "metformina" not in res_b.lower(), "Resumo de Carlos não pode ter Metformina"

    # Resumo Paciente Vazio
    r_resumo_vazio = requests.post(f"{BASE_URL}/api/resumo", json={"session_id": paciente_vazio})
    assert r_resumo_vazio.status_code == 200
    res_vazio = r_resumo_vazio.json()["resumo"]
    print(f"  [Resumo Vazio]: {res_vazio}")
    assert "não há registros" in res_vazio.lower() or "nenhum registro" in res_vazio.lower()
    print("  ✓ Resumo de paciente sem histórico validado.")

if __name__ == "__main__":
    try:
        test_matricula_validation()
        test_add_records_and_listing()
        test_chat_isolation()
        test_resumo_isolation()
        print("\n=======================================================")
        print("🎉 TODOS OS TESTES PASSARAM COM 100% DE SUCESSO! 🎉")
        print("=======================================================")
    except Exception as e:
        print(f"\n❌ FALHA NO TESTE: {e}")
        sys.exit(1)

