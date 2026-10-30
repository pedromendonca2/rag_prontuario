"""
Módulo de Persistência em Arquivos JSON.

Responsável pela serialização e desserialização de arquivos JSON no disco,
incluindo a base estática de prontuários médicos e a persistência incremental
de evoluções adicionadas em tempo de execução através da API.
"""
import json
import os


def salvar_json(dados, caminho):
    """
    Salva uma estrutura de dados Python em formato JSON no caminho especificado.

    Args:
        dados (dict | list): Dados a serem gravados em formato JSON.
        caminho (str): Caminho absoluto ou relativo do arquivo de destino.
    """
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"[persistencia] JSON salvo com sucesso em: {caminho}")


def carregar_json(caminho):
    """
    Carrega dados de um arquivo JSON caso o mesmo exista no disco.

    Args:
        caminho (str): Caminho do arquivo a ser lido.

    Returns:
        dict | list | None: Estrutura desserializada ou None se o arquivo não existir.
    """
    if not os.path.exists(caminho):
        return None
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_novo_registro_dinamico(registro, caminho):
    """
    Adiciona um novo registro clínico à lista persistente de dados dinâmicos.

    Essa função garante que qualquer evolução criada pelo médico via API
    seja gravada em disco para não se perder caso o ChromaDB precise ser
    reconstruído do zero.

    Args:
        registro (dict): Dicionário contendo dados da evolução médica (id, paciente_id, texto, data, etc.).
        caminho (str): Caminho do arquivo JSON de backup dinâmico.

    Returns:
        dict: O próprio registro adicionado.
    """
    registros = carregar_registros_dinamicos(caminho)
    registros.append(registro)
    salvar_json(registros, caminho)
    return registro


def carregar_registros_dinamicos(caminho):
    """
    Recupera a lista de registros adicionados dinamicamente em runtime.

    Args:
        caminho (str): Caminho do arquivo JSON.

    Returns:
        list: Lista de registros cadastrados, ou lista vazia se inexistente.
    """
    dados = carregar_json(caminho)
    if isinstance(dados, list):
        return dados
    return []
