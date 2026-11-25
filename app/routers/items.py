"""Endpoints de itens (filmes)."""

from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.routers.common import get_item_or_404, item_summary, normalize_genres
from app.services.recommender_service import service

router = APIRouter(prefix="/itens", tags=["Itens"])

NOT_FOUND = {404: {"model": schemas.Mensagem, "description": "Item não encontrado"}}


def _to_schema(db: Session, item: models.Item) -> schemas.Item:
    avg, count = db.execute(
        select(func.avg(models.Rating.rating), func.count()).where(models.Rating.item_id == item.id)
    ).one()
    return schemas.Item(
        id=item.id,
        titulo=item.title,
        generos=item.genres_list,
        ano=item.year,
        descricao=item.description or "",
        media_notas=round(float(avg), 2) if avg is not None else None,
        total_avaliacoes=count,
    )


@router.post("", response_model=schemas.Item, status_code=status.HTTP_201_CREATED)
def create_item(
    payload: schemas.ItemCriar, background: BackgroundTasks, db: Session = Depends(get_db)
):
    """Cadastra um item novo. Ele já pode ser recomendado na hora pela
    filtragem por conteúdo (gêneros + descrição); entra na colaborativa
    depois que receber avaliações suficientes e o modelo for retreinado."""
    item = models.Item(
        title=payload.titulo.strip(),
        genres="|".join(normalize_genres(payload.generos, allow_new=True)),
        year=payload.ano,
        description=payload.descricao.strip().lower(),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    service.refresh_content(db)
    if service.register_change():
        background.add_task(service.train)
    return _to_schema(db, item)


@router.get("", response_model=schemas.ListaItens)
def list_items(
    busca: str | None = Query(None, description="Trecho do título"),
    genero: str | None = Query(None, description="Ex.: Comedy"),
    limite: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Lista/busca itens do catálogo (útil para achar IDs para avaliar)."""
    q = select(models.Item)
    if busca:
        q = q.where(models.Item.title.ilike(f"%{busca}%"))
    if genero:
        q = q.where(models.Item.genres.ilike(f"%{genero}%"))
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(models.Item.id).limit(limite).offset(offset)).all()
    return schemas.ListaItens(total=total, itens=[item_summary(i) for i in rows])


@router.get("/{item_id}", response_model=schemas.Item, responses=NOT_FOUND)
def get_item(item_id: int, db: Session = Depends(get_db)):
    return _to_schema(db, get_item_or_404(db, item_id))


@router.get("/{item_id}/similares", response_model=list[schemas.ItemSimilar], responses=NOT_FOUND)
def similar_items(item_id: int, n: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)):
    """"Quem gostou deste também gostou de...". Usa a colaborativa quando o
    item tem avaliações suficientes e completa com similaridade de conteúdo."""
    get_item_or_404(db, item_id)
    sims = service.model.similar_items(item_id, n)
    items = {
        i.id: i for i in db.scalars(select(models.Item).where(models.Item.id.in_([s[0] for s in sims])))
    }
    return [
        schemas.ItemSimilar(item=item_summary(items[i]), similaridade=round(s, 4), fonte=src)
        for i, s, src in sims
        if i in items
    ]
