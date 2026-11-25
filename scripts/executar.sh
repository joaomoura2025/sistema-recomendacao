#!/usr/bin/env bash
# Sobe a API localmente (sem Docker). Rode ./scripts/instalar.sh antes.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source .venv/bin/activate
echo "Swagger: http://localhost:8000/docs"
exec uvicorn app.main:app --reload --port "${PORT:-8000}"
