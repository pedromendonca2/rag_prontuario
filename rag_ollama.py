import json
import requests

import config


def consultar_ollama(prompt, modelo=None, temperatura=0.3, max_tokens=1024):
    modelo = modelo or config.MODELO_LLM
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperatura,
            "num_predict": max_tokens,
        },
    }
    try:
        # Aumentado o timeout de 120 para 600 segundos (10 minutos)
        # Processar 12000 caracteres de contexto no RAG pode demorar muito em CPUs sem placa de vídeo.
        resposta = requests.post(config.OLLAMA_URL, json=payload, timeout=600)
        resposta.raise_for_status()
        return resposta.json()["response"].strip()
    except requests.exceptions.HTTPError as e:
        erro_msg = resposta.text if 'resposta' in locals() else str(e)
        return f"[ERRO] Falha na API do Ollama (404): {erro_msg}. Você já baixou o modelo rodando 'ollama pull {modelo}'?"
    except Exception as e:
        return f"[ERRO] Falha ao consultar Ollama: {e}"


def formatar_chunks_para_contexto(chunks):
    linhas = []
    for doc, meta, dist in chunks:
        data = meta.get("data", "")
        tipo = meta.get("tipo_evento", "")
        linhas.append(f"[{data}] ({tipo}) {doc}")
    return "\n\n".join(linhas)


def formatar_historico_para_prompt(historico):
    if not historico.listar():
        return "Nenhum historico anterior."
    linhas = []
    for turno in historico.listar():
        papel = "Medico" if turno["role"] == "user" else "Assistente"
        linhas.append(f"{papel}: {turno['content']}")
    return "\n".join(linhas)


def montar_prompt_resumo(contexto):
    return (
        "Você é um assistente clínico especializado em extração de dados de prontuários. "
        "Sua função é ESTRITAMENTE estruturar as informações solicitadas baseando-se APENAS no contexto fornecido. "
        "REGRAS CRÍTICAS:\n"
        "- Não faça interpretações, diagnósticos, suposições ou gere insights sobre a condição clínica ou prognóstico do paciente (ex: não preveja tempo de vida ou evolução).\n"
        "- Seja um mero relator dos fatos registrados. Se uma categoria não tiver informações no contexto, diga 'Informação não disponível'.\n\n"
        "Com base nas evoluções clínicas abaixo, gere um Resumo Clínico Estruturado contendo:\n"
        "1. Dados do paciente (idade, condição inicial)\n"
        "2. Resumo da evolução temporal (linha do tempo)\n"
        "3. Medicações utilizadas e histórico de doses\n"
        "4. Eventos adversos e intercorrências\n"
        "5. Resultados de exames relevantes\n"
        "6. Controle pressórico atual vs inicial\n"
        "7. Conduta atual e plano de seguimento\n\n"
        "Evoluções clínicas:\n"
        f"{contexto}\n\n"
        "Resumo Clínico Estruturado (Apenas fatos do prontuário):"
    )


def montar_prompt_rag(pergunta, chunks, historico):
    contexto = formatar_chunks_para_contexto(chunks)
    historico_str = formatar_historico_para_prompt(historico)

    return (
        "Você é um assistente clínico especializado em extração de dados de prontuários. "
        "Sua função é ESTRITAMENTE recuperar as informações solicitadas baseando-se APENAS no contexto fornecido e no histórico da conversa. "
        "REGRAS CRÍTICAS:\n"
        "- Não faça interpretações, diagnósticos, suposições ou gere insights sobre a condição clínica ou prognóstico do paciente (ex: não preveja tempo de vida ou evolução).\n"
        "- Seja um mero relator dos fatos registrados. Se a informação não estiver escrita textualmente no contexto, diga exatamente: 'Não encontrei essa informação nos registros disponíveis'.\n\n"
        "Contexto dos registros do paciente:\n"
        f"{contexto}\n\n"
        "Histórico da conversa:\n"
        f"{historico_str}\n\n"
        "Pergunta do médico:\n"
        f"{pergunta}\n\n"
        "Resposta (Apenas fatos do prontuário):"
    )


def gerar_resumo_clinico(indexador, modelo=None):
    print("[rag] Gerando resumo clinico...")

    num_docs = indexador.contar()
    todos_docs = indexador.buscar_por_texto("resumo completo do paciente", k=num_docs)

    chunks_texto = formatar_chunks_para_contexto(todos_docs)

    if len(chunks_texto) > 12000:
        chunks_texto = chunks_texto[:12000]
        print(f"[rag] Contexto truncado para ~12000 caracteres")

    prompt = montar_prompt_resumo(chunks_texto)
    print("[rag] Enviando para Ollama...")
    resposta = consultar_ollama(prompt, modelo=modelo, temperatura=0.2, max_tokens=2048)
    return resposta
