"""
Módulo de Gerenciamento de Memória Conversacional.

Implementa um buffer deslizante de histórico de mensagens para sustentar
contexto contínuo durante conversas clínicas entre o médico e o assistente RAG.
"""
import config


class HistoricoMemoria:
    """
    Mantém uma janela deslizante das últimas mensagens trocadas em uma sessão.

    Evita que o prompt exceda a janela de contexto da LLM descartando as
    mensagens mais antigas quando o limite máximo configurado é atingido.
    """

    def __init__(self, limite=None):
        """
        Inicializa o buffer de memória.

        Args:
            limite (int, opcional): Quantidade máxima de turnos conversacionais
                a manter. Se omitido, utiliza `config.LIMITE_HISTORICO`.
        """
        self.limite = limite or config.LIMITE_HISTORICO
        self._turnos = []

    def adicionar(self, role: str, content: str):
        """
        Adiciona uma nova mensagem (do médico ou do assistente) ao histórico.

        Args:
            role (str): Papel do autor ('user' para médico, 'assistant' para IA).
            content (str): Texto da mensagem.
        """
        self._turnos.append({"role": role, "content": content})
        if len(self._turnos) > self.limite:
            # Mantém apenas os últimos N turnos mais recentes
            self._turnos = self._turnos[-self.limite:]

    def listar(self) -> list:
        """
        Retorna uma cópia da lista ordenada de turnos conversacionais.

        Returns:
            list: Lista de dicionários no formato [{"role": str, "content": str}, ...].
        """
        return list(self._turnos)

    def limpar(self):
        """Esvazia todo o histórico de mensagens da sessão atual."""
        self._turnos.clear()

    def ultimo_usuario(self) -> str:
        """
        Retorna o texto da última mensagem enviada pelo usuário (médico).

        Returns:
            str: Conteúdo da mensagem ou string vazia se não houver.
        """
        for turno in reversed(self._turnos):
            if turno["role"] == "user":
                return turno["content"]
        return ""
