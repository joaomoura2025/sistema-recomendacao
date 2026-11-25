#!/usr/bin/env bash
# Instalação local (sem Docker): cria o ambiente virtual, instala as
# dependências, baixa o MovieLens e carrega no banco SQLite.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt

python -m scripts.download_movielens
python -m scripts.seed

echo
echo "Pronto! Para subir a API:  ./scripts/executar.sh"
