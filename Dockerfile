# ---- Imagem da API de recomendação ----
FROM python:3.13-slim

# Python sem buffer de saída (logs aparecem na hora) e sem arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependências primeiro: assim o Docker reaproveita esta camada em cache
# quando só o código muda.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini ./

# Usuário sem privilégios de root
RUN useradd --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser /app \
    && chmod +x scripts/entrypoint.sh
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude')" || exit 1

ENTRYPOINT ["scripts/entrypoint.sh"]
