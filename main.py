import argparse
import os
import sys

import config
from dados_sinteticos import gerar_evolucoes
from persistencia import salvar_json, carregar_json
from indexador import IndexadorVectorial
from rag_ollama import gerar_resumo_clinico
from cli import cli_loop


def main():
    parser = argparse.ArgumentParser(
        description="RAG Prontuario Medico - Pipeline de processamento local de prontuarios"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Forca reindexacao de todos os documentos no banco vetorial",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=True,
        help="Inicia sessao de chat interativo (padrao)",
    )
    parser.add_argument(
        "--resumo",
        action="store_true",
        help="Gera resumo clinico estruturado via LLM e exibe no terminal",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        help=f"Modelo Ollama (padrao: {config.MODELO_LLM})",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Executa sem iniciar chat (usado com --resumo)",
    )

    args = parser.parse_args()

    modelo = args.modelo or config.MODELO_LLM

    dados_json = carregar_json(config.ARQUIVO_JSON)
    if dados_json is None:
        print("[main] Gerando dados sinteticos...")
        dados_json = gerar_evolucoes()
        salvar_json(dados_json, config.ARQUIVO_JSON)
        print(f"[main] {len(dados_json)} registros gerados")
    else:
        print(f"[main] JSON carregado: {len(dados_json)} registros")

    indexador = IndexadorVectorial()

    if args.reindex or indexador.contar() == 0:
        print("[main] Indexando documentos no banco vetorial...")
        indexador.indexar(dados_json)
    else:
        print(f"[main] Banco vetorial ja possui {indexador.contar()} documentos")

    if args.resumo:
        print("\n[main] ===== GERANDO RESUMO CLINICO =====")
        resumo = gerar_resumo_clinico(indexador, modelo=modelo)
        print("\n" + "=" * 60)
        print("RESUMO CLINICO ESTRUTURADO")
        print("=" * 60)
        print(resumo)
        print("=" * 60)

    if args.no_chat:
        return

    if args.chat:
        print("\n[main] ===== INICIANDO CHAT INTERATIVO =====")
        cli_loop(indexador, modelo_llm=modelo)


if __name__ == "__main__":
    main()
