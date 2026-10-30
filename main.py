"""
Ponto de Entrada e Orquestrador Principal do Sistema RAG.

Permite executar o sistema em diferentes modos de operação:
1. Web API: `python main.py --api` (sobe o servidor FastAPI na porta 8000).
2. Worker AMQP: `python main.py --worker` (sobe o consumidor RabbitMQ).
3. Chat Interativo no Terminal: `python main.py` ou `python main.py --chat`.
4. Geração de Resumo Direto: `python main.py --resumo --no-chat`.
5. Reindexação Vetorial: `python main.py --reindex`.
"""
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
    """Analisa argumentos de linha de comando e despacha a execução para o modo solicitado."""
    parser = argparse.ArgumentParser(
        description="RAG Prontuário Médico - Pipeline local de inteligência clínica e busca vetorial"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Força reindexação completa de todos os documentos no banco vetorial ChromaDB",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Inicia sessão de chat interativo pelo terminal (padrão se nenhum outro modo for especificado)",
    )
    parser.add_argument(
        "--resumo",
        action="store_true",
        help="Gera o resumo clínico estruturado diretamente via LLM e exibe no terminal",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Inicia o servidor web REST FastAPI (porta 8000)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Inicia o worker consumidor assíncrono do RabbitMQ",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        help=f"Nome do modelo LLM no Ollama (padrão: {config.MODELO_LLM})",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Executa a tarefa (ex: --resumo) e encerra sem abrir o prompt de chat",
    )

    args = parser.parse_args()

    # Modo 1: API REST FastAPI
    if args.api:
        import uvicorn
        print("\n[main] ===== INICIANDO FASTAPI (PORTA 8000) =====")
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
        return

    # Modo 2: Worker de IA (RabbitMQ Consumer)
    if args.worker:
        import worker
        import asyncio
        print("\n[main] ===== INICIANDO WORKER RABBITMQ (FILA RPC) =====")
        asyncio.run(worker.main())
        return

    # Modo 3: CLI / Terminal direto
    if not (args.resumo or args.worker or args.api):
        args.chat = True

    modelo = args.modelo or config.MODELO_LLM

    # Carga dos dados de prontuário
    dados_json = carregar_json(config.ARQUIVO_JSON)
    if dados_json is None:
        print("[main] Arquivo de prontuário não encontrado. Gerando dados sintéticos iniciais...")
        dados_json = gerar_evolucoes()
        salvar_json(dados_json, config.ARQUIVO_JSON)
        print(f"[main] {len(dados_json)} registros gerados com sucesso.")
    else:
        print(f"[main] JSON de prontuários carregado: {len(dados_json)} registros disponíveis.")

    indexador = IndexadorVectorial()

    # Reindexação se solicitada ou se o banco estiver vazio
    if args.reindex or indexador.contar() == 0:
        print("[main] Indexando documentos no banco vetorial ChromaDB...")
        indexador.indexar(dados_json)
        from persistencia import carregar_registros_dinamicos
        dinamicos = carregar_registros_dinamicos(config.ARQUIVO_DINAMICO)
        if dinamicos:
            print(f"[main] Reindexando {len(dinamicos)} registros dinâmicos persistidos...")
            for reg in dinamicos:
                indexador.adicionar_registro(
                    texto=reg["texto"],
                    paciente_id=reg["paciente_id"],
                    data=reg.get("data"),
                    tipo_evento=reg.get("tipo_evento"),
                    profissional=reg.get("profissional"),
                    id_registro=reg.get("id"),
                )
    else:
        print(f"[main] Banco vetorial já populado com {indexador.contar()} documentos.")

    if args.resumo:
        print("\n[main] ===== GERANDO RESUMO CLÍNICO ESTRUTURADO =====")
        resumo = gerar_resumo_clinico(indexador, modelo=modelo)
        print("\n" + "=" * 60)
        print("RESUMO CLÍNICO ESTRUTURADO")
        print("=" * 60)
        print(resumo)
        print("=" * 60)

    if args.no_chat:
        return

    if args.chat:
        print("\n[main] ===== INICIANDO CHAT INTERATIVO NO TERMINAL =====")
        cli_loop(indexador, modelo_llm=modelo)


if __name__ == "__main__":
    main()
