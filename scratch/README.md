# Scripts Utilitários, Benchmarks e Diagnósticos (`scratch/`)

Este diretório contém ferramentas auxiliares desenvolvidas para validação clínica, testes de carga, anonimização e verificação do motor RAG.

---

## 🛠️ Relação de Scripts

### 1. `gerar_prontuario_100.py`
- **Finalidade:** Gera e indexa diretamente no ChromaDB um prontuário sintético com **100 evoluções cronológicas realistas** cobrindo o período de janeiro de 2024 a outubro de 2026.
- **Matrícula de Teste:** `paciente_100_evol`
- **Uso:** Utilizado para aferir o tempo de resposta do modelo LLM (`llama3:latest`) e do banco vetorial em cenários de prontuários extensos de pacientes crônicos ou com múltiplas internações.
- **Como executar:**
  ```bash
  docker exec -it rag_api python scratch/gerar_prontuario_100.py
  ```

---

### 2. `anonimizar_prontuarios.py`
- **Finalidade:** Motor de conformidade com a LGPD e HIPAA. Aplica expressões regulares para desidentificar prontuários reais, substituindo nomes de acompanhantes, parentes, hospitais específicos e identificadores por descritores genéricos.
- **Como executar:**
  ```bash
  docker exec -it rag_api python scratch/anonimizar_prontuarios.py
  ```

---

### 3. `test_multi_patient.py`
- **Finalidade:** Bateria de testes automatizados para verificar o **isolamento estrito entre matrículas de pacientes**.
- **O que valida:**
  - Garante que perguntas feitas na sessão do paciente `A` nunca retornem dados, prescrições ou diagnósticos pertencentes ao paciente `B`.
  - Testa filtros de metadados no ChromaDB (`{"paciente_id": ...}`).
- **Como executar:**
  ```bash
  docker exec -it rag_api python scratch/test_multi_patient.py
  ```

---

### 4. `test_real_patients_rag.py`
- **Finalidade:** Bateria de testes clínicos com perguntas reais formuladas sobre os prontuários de Pronto-Socorro e Enfermaria sanitizados (`ps_01` a `ps_10`, `enf_11` a `enf_20`).
- **Como executar:**
  ```bash
  docker exec -it rag_api python scratch/test_real_patients_rag.py
  ```

---

> [!IMPORTANT]
> Nunca versione dados com identificadores de pacientes reais. Todos os scripts deste diretório devem operar exclusivamente sobre dados sanitizados ou sintéticos.
