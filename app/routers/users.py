"""Endpoints de usuários, avaliações (preferências) e recomendações."""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.recommender import Strategy
from app.routers.common import get_item_or_404, get_user_or_404, item_summary, normalize_genres
from app.services.recommender_service import service

router = APIRouter(prefix="/usuarios", tags=["Usuários"])

NOT_FOUND = {404: {"model": schemas.Mensagem, "description": "Usuário ou item não encontrado"}}


def _to_schema(db: Session, user: models.User) -> schemas.Usuario:
    total = db.scalar(select(func.count()).where(models.Rating.user_id == user.id))
    return schemas.Usuario(
        id=user.id,
        nome=user.name,
        generos_favoritos=user.favorite_genres_list,
        total_avaliacoes=total or 0,
        criado_em=user.created_at,
    )


def _after_change(background: BackgroundTasks, n: int = 1) -> None:
    """Registra a alteração e, se acumulou o suficiente, agenda retreino."""
    if service.register_change(n):
        background.add_task(service.train)


# ------------------------------------------------------------------ usuários
@router.post("", response_model=schemas.Usuario, status_code=status.HTTP_201_CREATED)
def create_user(payload: schemas.UsuarioCriar, db: Session = Depends(get_db)):
    """Cria um usuário. Os gêneros favoritos são opcionais, mas melhoram as
    primeiras recomendações (antes de ele avaliar qualquer filme)."""
    user = models.User(
        name=payload.nome,
        favorite_genres="|".join(normalize_genres(payload.generos_favoritos)),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _to_schema(db, user)


@router.get("/{user_id}", response_model=schemas.Usuario, responses=NOT_FOUND)
def get_user(user_id: int, db: Session = Depends(get_db)):
    return _to_schema(db, get_user_or_404(db, user_id))


@router.patch("/{user_id}/preferencias", response_model=schemas.Usuario, responses=NOT_FOUND)
def update_preferences(
    user_id: int, payload: schemas.PreferenciasAtualizar, db: Session = Depends(get_db)
):
    """Atualiza nome e/ou gêneros favoritos. Para mudar o gosto do usuário
    pelas notas, use os endpoints de avaliações."""
    user = get_user_or_404(db, user_id)
    if payload.nome is not None:
        user.name = payload.nome
    if payload.generos_favoritos is not None:
        user.favorite_genres = "|".join(normalize_genres(payload.generos_favoritos))
    db.commit()
    return _to_schema(db, user)


# ---------------------------------------------------------------- avaliações
@router.get("/{user_id}/avaliacoes", response_model=list[schemas.Avaliacao], responses=NOT_FOUND)
def list_ratings(user_id: int, db: Session = Depends(get_db)):
    get_user_or_404(db, user_id)
    rows = db.execute(
        select(models.Rating, models.Item.title)
        .join(models.Item, models.Item.id == models.Rating.item_id)
        .where(models.Rating.user_id == user_id)
        .order_by(models.Rating.timestamp.desc())
    ).all()
    return [
        schemas.Avaliacao(item_id=r.item_id, titulo=title, nota=r.rating, data=r.timestamp)
        for r, title in rows
    ]


def _upsert(db: Session, user_id: int, item_id: int, nota: float) -> bool:
    """Cria ou atualiza uma avaliação. Devolve True se foi criada."""
    rating = db.scalar(
        select(models.Rating).where(
            models.Rating.user_id == user_id, models.Rating.item_id == item_id
        )
    )
    now = datetime.now(timezone.utc)
    if rating is None:
        db.add(models.Rating(user_id=user_id, item_id=item_id, rating=nota, timestamp=now))
        return True
    rating.rating = nota
    rating.timestamp = now
    return False


@router.put(
    "/{user_id}/avaliacoes/{item_id}",
    response_model=schemas.Avaliacao,
    responses={**NOT_FOUND, 201: {"model": schemas.Avaliacao, "description": "Avaliação criada"}},
)
def rate_item(
    user_id: int,
    item_id: int,
    payload: schemas.NotaEntrada,
    response: Response,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Cria ou atualiza a nota do usuário para um item (0.5 a 5, de 0.5 em 0.5).
    O efeito nas recomendações é imediato: não precisa retreinar."""
    get_user_or_404(db, user_id)
    item = get_item_or_404(db, item_id)
    created = _upsert(db, user_id, item_id, payload.nota)
    db.commit()
    _after_change(background)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return schemas.Avaliacao(
        item_id=item_id, titulo=item.title, nota=payload.nota, data=datetime.now(timezone.utc)
    )


@router.post("/{user_id}/avaliacoes", response_model=schemas.ResultadoLote, responses=NOT_FOUND)
def rate_many(
    user_id: int,
    payload: schemas.AvaliacoesLote,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Envia várias avaliações de uma vez (útil para o "onboarding")."""
    get_user_or_404(db, user_id)
    for a in payload.avaliacoes:
        get_item_or_404(db, a.item_id)
    created = sum(_upsert(db, user_id, a.item_id, a.nota) for a in payload.avaliacoes)
    db.commit()
    _after_change(background, len(payload.avaliacoes))
    return schemas.ResultadoLote(criadas=created, atualizadas=len(payload.avaliacoes) - created)


@router.delete(
    "/{user_id}/avaliacoes/{item_id}", status_code=status.HTTP_204_NO_CONTENT, responses=NOT_FOUND
)
def delete_rating(
    user_id: int, item_id: int, background: BackgroundTasks, db: Session = Depends(get_db)
):
    get_user_or_404(db, user_id)
    rating = db.scalar(
        select(models.Rating).where(
            models.Rating.user_id == user_id, models.Rating.item_id == item_id
        )
    )
    if rating is None:
        get_item_or_404(db, item_id)
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Avaliação não encontrada")
    db.delete(rating)
    db.commit()
    _after_change(background)


# ------------------------------------------------------------ recomendações
@router.get(
    "/{user_id}/recomendacoes", response_model=schemas.RespostaRecomendacoes, responses=NOT_FOUND
)
def recommend(
    user_id: int,
    n: int = Query(10, ge=1, le=100, description="Quantidade de recomendações"),
    estrategia: Strategy = Query(Strategy.HYBRID, description="Algoritmo usado"),
    genero: str | None = Query(None, description="Filtra por um gênero (ex.: Comedy)"),
    diversidade: float = Query(
        0.0, ge=0.0, le=1.0,
        description="0 = só relevância; valores maiores evitam recomendar filmes parecidos entre si (MMR)",
    ),
    db: Session = Depends(get_db),
):
    """Recomenda itens que o usuário ainda não avaliou, com o motivo de cada
    recomendação. Funciona também para usuários novos (cold start): usa os
    gêneros favoritos e, sem eles, os filmes mais populares."""
    user = get_user_or_404(db, user_id)
    if genero:
        genero = normalize_genres([genero])[0]

    ratings = dict(
        db.execute(
            select(models.Rating.item_id, models.Rating.rating).where(
                models.Rating.user_id == user_id
            )
        ).all()
    )
    model = service.model  # referência local: um retreino no meio não afeta esta requisição
    recs = model.recommend(
        ratings,
        favorite_genres=user.favorite_genres_list,
        n=n,
        strategy=estrategia,
        genre=genero,
        diversity=diversidade,
    )

    items = {
        i.id: i
        for i in db.scalars(select(models.Item).where(models.Item.id.in_([r.item_id for r in recs])))
    }
    alpha = model.alpha(ratings) if estrategia == Strategy.HYBRID else (
        1.0 if estrategia == Strategy.COLLABORATIVE else 0.0
    )
    return schemas.RespostaRecomendacoes(
        usuario_id=user_id,
        estrategia=estrategia,
        total_avaliacoes=len(ratings),
        peso_colaborativa=round(alpha, 3),
        recomendacoes=[
            schemas.Recomendacao(
                item=item_summary(items[r.item_id]),
                score=r.score,
                componentes=schemas.Componentes(
                    colaborativa=r.collaborative, conteudo=r.content, popularidade=r.popularity
                ),
                motivo=r.reason,
            )
            for r in recs
            if r.item_id in items
        ],
    )
