"""Aplicação FastAPI: Sistema de Recomendação de Filmes.

Rodar localmente:
    uvicorn app.main:app --reload
Documentação interativa (Swagger): http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from app.database import Base, SessionLocal, engine
from app.routers import items, model, users
from app.services.recommender_service import service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DESCRIPTION = """
API de recomendação de filmes (dataset **MovieLens**) com modelo **híbrido**:

* **Filtragem colaborativa** item-item (kNN com shrinkage): "quem gostou de X também gostou de Y";
* **Filtragem baseada em conteúdo**: gêneros + tags (TF-IDF) + década;
* **Popularidade** (média bayesiana) para usuários novos.

O peso de cada parte se adapta ao histórico do usuário: α = n / (n + K).
Cada recomendação vem com o **motivo** em português.

**Fluxo sugerido:** `POST /usuarios` → `GET /itens?busca=matrix` →
`PUT /usuarios/{id}/avaliacoes/{item_id}` → `GET /usuarios/{id}/recomendacoes`.
"""

TAGS = [
    {"name": "Usuários", "description": "Cadastro, preferências, avaliações e recomendações"},
    {"name": "Itens", "description": "Catálogo de filmes e itens similares"},
    {"name": "Modelo", "description": "Status e retreino do modelo"},
    {"name": "Sistema", "description": "Saúde da aplicação e metadados"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    service.train()
    yield


app = FastAPI(
    title="Sistema de Recomendação de Filmes",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
)
app.include_router(users.router)
app.include_router(items.router)
app.include_router(model.router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/saude", tags=["Sistema"])
def health():
    """Verifica se a API e o banco estão no ar (usado pelo healthcheck do Docker)."""
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
    return {"status": "ok", "modelo_treinado": service.trained_at is not None}


@app.get("/generos", tags=["Sistema"], response_model=list[str])
def list_genres():
    """Gêneros disponíveis no catálogo (para usar em generos_favoritos)."""
    return service.model.known_genres()
