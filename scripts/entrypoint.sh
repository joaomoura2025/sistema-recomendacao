#!/bin/sh
# Ponto de entrada do container:
#   1. carrega o MovieLens no banco (só na primeira vez)
#   2. sobe a API
set -e

if [ "${SEED_ON_STARTUP:-true}" = "true" ]; then
    python -m scripts.seed
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
