import config


class HistoricoMemoria:
    def __init__(self, limite=None):
        self.limite = limite or config.LIMITE_HISTORICO
        self._turnos = []

    def adicionar(self, role, content):
        self._turnos.append({"role": role, "content": content})
        if len(self._turnos) > self.limite:
            self._turnos = self._turnos[-self.limite:]

    def listar(self):
        return list(self._turnos)

    def limpar(self):
        self._turnos.clear()

    def ultimo_usuario(self):
        for turno in reversed(self._turnos):
            if turno["role"] == "user":
                return turno["content"]
        return ""
