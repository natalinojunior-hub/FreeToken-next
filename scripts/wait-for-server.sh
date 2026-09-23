#!/usr/bin/env bash
# Espera o servidor iniciar usando ping event-driven. Fails fast se o processo morrer.

PORT=${1:-8081}
PID=${2:-}
TIMEOUT=${3:-180}
START=$(date +%s)

echo "Aguardando servidor na porta $PORT..."

while true; do
    if curl -s http://127.0.0.1:$PORT/health | grep -Eq '"status"[[:space:]]*:[[:space:]]*"(ok|serving)"'; then
        echo "Servidor ONLINE!"
        exit 0
    fi
    
    if [ -n "$PID" ] && ! kill -0 $PID 2>/dev/null; then
        echo "ERRO: O processo do servidor (PID $PID) morreu antes de ficar online!"
        exit 1
    fi
    
    NOW=$(date +%s)
    if [ $((NOW - START)) -gt $TIMEOUT ]; then
        echo "ERRO: Timeout de $TIMEOUT segundos aguardando o servidor."
        exit 1
    fi
    
    sleep 0.5
done
