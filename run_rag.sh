#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VENV="$DIR/venv"

echo "=== Iniciando RAG Prontuario ==="
source "$VENV/bin/activate"

echo "[1/4] Encerrando instâncias anteriores..."
# Libera porta 8000 se estiver em uso
fuser -k 8000/tcp 2>/dev/null || true
pkill -f "(worker\.py|api:app|main\.py --worker|main\.py --api)" 2>/dev/null || true
sleep 1

echo "[2/4] Iniciando worker.py com setsid..."
setsid "$VENV/bin/python" -u "$DIR/worker.py" > "$DIR/worker.log" 2>&1 < /dev/null &
WORKER_PID=$!
echo "Worker iniciado com PID $WORKER_PID"

echo "[3/4] Iniciando FastAPI via uvicorn com setsid..."
setsid "$VENV/bin/uvicorn" api:app --host 0.0.0.0 --port 8000 --app-dir "$DIR" > "$DIR/api.log" 2>&1 < /dev/null &
API_PID=$!
echo "API iniciada com PID $API_PID"

echo "[4/4] Validando inicializacao dos servicos..."
SUCCESS=0
for i in {1..15}; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "[ERRO] Worker morreu durante a inicializacao!"
        cat "$DIR/worker.log"
        exit 1
    fi
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "[ERRO] API uvicorn morreu durante a inicializacao!"
        cat "$DIR/api.log"
        exit 1
    fi

    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        SUCCESS=1
        break
    fi
    sleep 1
done

if [ $SUCCESS -eq 1 ]; then
    echo "=== RAG Services inicializados e saudaveis! ==="
    echo "  - Worker PID: $WORKER_PID"
    echo "  - API PID: $API_PID (http://localhost:8000/docs)"
else
    echo "[ERRO] Timeout aguardando API responder na porta 8000."
    echo "Log API:"
    cat "$DIR/api.log"
    exit 1
fi
