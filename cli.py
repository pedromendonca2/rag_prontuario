import os
import config
from rag_ollama import consultar_ollama, montar_prompt_rag, gerar_resumo_clinico
from chat_memoria import HistoricoMemoria


LINHA_SEPARADORA = "-" * 60


def cli_loop(indexador, modelo_llm=None):
    modelo_llm = modelo_llm or config.MODELO_LLM
    historico = HistoricoMemoria()

    print(LINHA_SEPARADORA)
    print("SISTEMA RAG - PRONTUARIO MEDICO")
    print(f"Modelo LLM: {modelo_llm}")
    print(f"Documentos indexados: {indexador.contar()}")
    print(LINHA_SEPARADORA)
    print("Comandos disponiveis:")
    print("  /resumo   - Gerar resumo clinico estruturado")
    print("  /limpar   - Limpar historico da conversa e a tela")
    print("  /sair     - Encerrar sessao")
    print("  <livre>   - Faca uma pergunta sobre o prontuario")
    print(LINHA_SEPARADORA)

    while True:
        try:
            entrada = input("\nVoce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando sessao.")
            break

        if not entrada:
            continue

        if entrada.lower() == "/sair":
            print("Sessao encerrada.")
            break

        if entrada.lower() == "/limpar":
            historico.limpar()
            os.system('cls' if os.name == 'nt' else 'clear')
            print("[sistema] Historico da conversa limpo.")
            continue

        if entrada.lower() == "/resumo":
            print("\n[rag] Gerando resumo clinico...\n")
            try:
                resposta = gerar_resumo_clinico(indexador, modelo=modelo_llm)
                print(LINHA_SEPARADORA)
                print("RESUMO CLINICO ESTRUTURADO:")
                print(LINHA_SEPARADORA)
                print(resposta)
                print(LINHA_SEPARADORA)
            except Exception as e:
                print(f"[ERRO] Falha ao gerar resumo: {e}")
            continue

        print(f"\n[sistema] Buscando contexto relevante...")
        chunks = indexador.buscar_por_texto(entrada)

        if not chunks:
            print("[sistema] Nenhum contexto relevante encontrado.")
            continue

        print(f"[sistema] {len(chunks)} trechos recuperados.")

        historico.adicionar("user", entrada)
        prompt = montar_prompt_rag(entrada, chunks, historico)

        print("[sistema] Consultando LLM...")
        try:
            resposta = consultar_ollama(prompt, modelo=modelo_llm)
            historico.adicionar("assistant", resposta)

            print(f"\nAssistente:\n{resposta}")
        except Exception as e:
            print(f"[ERRO] Falha ao consultar LLM: {e}")
