"""
Módulo de Interface Interativa em Linha de Comando (CLI).

Permite interagir com o sistema RAG diretamente pelo terminal do desenvolvedor
sem necessidade de subir a API REST ou a interface gráfica Angular.
"""
import os
import config
from rag_ollama import consultar_ollama, montar_prompt_rag, gerar_resumo_clinico
from chat_memoria import HistoricoMemoria


LINHA_SEPARADORA = "-" * 60


def cli_loop(indexador, modelo_llm=None):
    """
    Loop de leitura e resposta interativo no terminal.

    Comandos Especiais:
    - /resumo: Gera o Resumo Clínico Estruturado do prontuário.
    - /limpar: Reseta a memória conversacional e limpa a tela.
    - /sair: Encerra a execução da CLI.

    Args:
        indexador (IndexadorVectorial): Instância ativa de acesso ao ChromaDB.
        modelo_llm (str, opcional): Nome do modelo no Ollama.
    """
    modelo_llm = modelo_llm or config.MODELO_LLM
    historico = HistoricoMemoria()

    print(LINHA_SEPARADORA)
    print("SISTEMA RAG - PRONTUÁRIO MÉDICO (MODO CLI)")
    print(f"Modelo LLM: {modelo_llm}")
    print(f"Documentos indexados: {indexador.contar()}")
    print(LINHA_SEPARADORA)
    print("Comandos disponíveis:")
    print("  /resumo   - Gerar resumo clínico estruturado")
    print("  /limpar   - Limpar histórico da conversa e a tela")
    print("  /sair     - Encerrar sessão")
    print("  <livre>   - Faça uma pergunta clínica sobre o prontuário")
    print(LINHA_SEPARADORA)

    while True:
        try:
            entrada = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando sessão.")
            break

        if not entrada:
            continue

        if entrada.lower() == "/sair":
            print("Sessão encerrada.")
            break

        if entrada.lower() == "/limpar":
            historico.limpar()
            os.system('cls' if os.name == 'nt' else 'clear')
            print("[sistema] Histórico da conversa limpo.")
            continue

        if entrada.lower() == "/resumo":
            print("\n[rag] Gerando resumo clínico estruturado...\n")
            try:
                resposta = gerar_resumo_clinico(indexador, modelo=modelo_llm)
                print(LINHA_SEPARADORA)
                print("RESUMO CLÍNICO ESTRUTURADO:")
                print(LINHA_SEPARADORA)
                print(resposta)
                print(LINHA_SEPARADORA)
            except Exception as e:
                print(f"[ERRO] Falha ao gerar resumo: {e}")
            continue

        print("\n[sistema] Buscando contexto clínico relevante no banco vetorial...")
        chunks = indexador.buscar_por_texto(entrada)

        if not chunks:
            print("[sistema] Nenhum contexto relevante encontrado.")
            continue

        print(f"[sistema] {len(chunks)} trechos clínicos recuperados.")

        historico.adicionar("user", entrada)
        prompt = montar_prompt_rag(entrada, chunks, historico)

        print(f"[sistema] Consultando LLM ({modelo_llm})...")
        try:
            resposta = consultar_ollama(prompt, modelo=modelo_llm)
            historico.adicionar("assistant", resposta)

            print(f"\nAssistente:\n{resposta}")
        except Exception as e:
            print(f"[ERRO] Falha ao consultar LLM: {e}")
