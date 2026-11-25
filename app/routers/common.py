"""Funções auxiliares compartilhadas pelos routers."""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.services.recommender_service import service


def get_user_or_404(db: Session, user_id: int) -> models.User:
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Usuário {user_id} não encontrado")
    return user


def get_item_or_404(db: Session, item_id: int) -> models.Item:
    item = db.get(models.Item, item_id)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Item {item_id} não encontrado")
    return item


def normalize_genres(genres: list[str], allow_new: bool = False) -> list[str]:
    """Aceita gêneros sem diferenciar maiúsculas ("sci-fi" -> "Sci-Fi").
    Para usuários, só aceita gêneros que existem no catálogo; para itens
    novos (allow_new=True), aceita gêneros novos."""
    known = {g.lower(): g for g in service.model.known_genres()}
    out, unknown = [], []
    for g in genres:
        g = g.strip()
        if not g:
            continue
        canon = known.get(g.lower())
        if canon:
            out.append(canon)
        elif allow_new or not known:
            out.append(g)
        else:
            unknown.append(g)
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Gênero(s) desconhecido(s): {', '.join(unknown)}. "
            f"Disponíveis: {', '.join(sorted(known.values()))}",
        )
    return list(dict.fromkeys(out))  # remove repetidos mantendo a ordem


def item_summary(item: models.Item) -> schemas.ItemResumo:
    return schemas.ItemResumo(id=item.id, titulo=item.title, generos=item.genres_list, ano=item.year)
