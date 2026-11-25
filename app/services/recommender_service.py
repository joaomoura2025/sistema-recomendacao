"""Ponte entre o banco de dados e o modelo de recomendação.

Mantém UMA instância do modelo em memória (compartilhada entre as
requisições). O retreino monta um modelo novo e só depois troca a
referência, então as requisições nunca veem um modelo pela metade.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Item, Rating
from app.recommender import HybridRecommender

log = logging.getLogger(__name__)


def load_items(db: Session) -> pd.DataFrame:
    rows = db.execute(
        select(Item.id, Item.title, Item.genres, Item.year, Item.description)
    ).all()
    return pd.DataFrame(rows, columns=["item_id", "title", "genres", "year", "description"])


def load_ratings(db: Session) -> pd.DataFrame:
    rows = db.execute(select(Rating.user_id, Rating.item_id, Rating.rating)).all()
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])


class RecommenderService:
    def __init__(self) -> None:
        self.model = self._new_model()
        self.trained_at: datetime | None = None
        self.train_seconds: float | None = None
        self.n_ratings = 0
        self.pending_changes = 0
        self._lock = threading.Lock()
        self._training = threading.Lock()

    @staticmethod
    def _new_model() -> HybridRecommender:
        return HybridRecommender(
            knn_neighbors=settings.knn_neighbors,
            knn_shrinkage=settings.knn_shrinkage,
            min_item_ratings=settings.min_item_ratings,
            hybrid_k=settings.hybrid_k,
            popularity_weight=settings.popularity_weight,
            content_popularity_mix=settings.content_popularity_mix,
        )

    # ----------------------------------------------------------------- treino
    def train(self, db: Session | None = None) -> None:
        """Treino completo (colaborativa + conteúdo + popularidade)."""
        with self._training:  # evita dois treinos ao mesmo tempo
            own = db is None
            db = db or SessionLocal()
            try:
                t0 = time.perf_counter()
                items, ratings = load_items(db), load_ratings(db)
                model = self._new_model()
                if not items.empty:
                    model.fit(items, ratings)
                elapsed = time.perf_counter() - t0
            finally:
                if own:
                    db.close()

            with self._lock:
                self.model = model
                self.trained_at = datetime.now(timezone.utc)
                self.train_seconds = round(elapsed, 3)
                self.n_ratings = len(ratings)
                self.pending_changes = 0
            log.info("Modelo treinado: %d itens, %d avaliações em %.2fs", len(items), len(ratings), elapsed)

    def refresh_content(self, db: Session) -> None:
        """Atualiza só o modelo de conteúdo (usado quando entra item novo)."""
        items = load_items(db)
        with self._lock:
            self.model.fit_content(items)

    def register_change(self, n: int = 1) -> bool:
        """Conta alterações desde o último treino. Devolve True quando já é
        hora de retreinar em segundo plano."""
        with self._lock:
            self.pending_changes += n
            due = settings.retrain_every > 0 and self.pending_changes >= settings.retrain_every
        return due

    def status(self) -> dict:
        m = self.model
        return {
            "treinado_em": self.trained_at,
            "duracao_treino_s": self.train_seconds,
            "itens": m.content.n_items,
            "itens_colaborativa": m.cf.n_items,
            "avaliacoes_no_treino": self.n_ratings,
            "alteracoes_pendentes": self.pending_changes,
            "retreino_automatico_a_cada": settings.retrain_every,
            "parametros": {
                "knn_vizinhos": m.knn_neighbors,
                "knn_shrinkage": m.knn_shrinkage,
                "min_avaliacoes_item": m.min_item_ratings,
                "hybrid_k": m.hybrid_k,
                "peso_popularidade": m.popularity_weight,
                "mistura_conteudo_popularidade": m.content_popularity_mix,
            },
        }


# Instância única usada pela aplicação
service = RecommenderService()
