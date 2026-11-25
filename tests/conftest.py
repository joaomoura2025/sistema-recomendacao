"""Fixtures compartilhadas.

Os testes usam um catálogo pequeno e "óbvio", onde dá para saber de
antemão o que o modelo deveria recomendar:

- filmes 1-4: ficção científica; 5-8: romance; 9-12: comédia
- usuários 1-10 adoram sci-fi e odeiam romance; 11-20, o contrário
- todo mundo dá nota média para comédia
"""

import os
import tempfile
from pathlib import Path

# O banco precisa ser definido ANTES de importar a aplicação
_TMP = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ.setdefault("RETRAIN_EVERY", "0")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models import Item, Rating, User  # noqa: E402

SCIFI, ROMANCE, COMEDY = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]


def catalog() -> pd.DataFrame:
    rows = []
    for i in SCIFI:
        rows.append((i, f"Space Movie {i}", "Sci-Fi|Action", 1990 + i, "space aliens spaceship"))
    for i in ROMANCE:
        rows.append((i, f"Love Movie {i}", "Romance|Drama", 2000 + i, "love story wedding"))
    for i in COMEDY:
        rows.append((i, f"Funny Movie {i}", "Comedy", 2010 + i, "funny jokes"))
    return pd.DataFrame(rows, columns=["item_id", "title", "genres", "year", "description"])


def ratings_df() -> pd.DataFrame:
    rows = []
    for u in range(1, 21):
        scifi_fan = u <= 10
        for i in SCIFI:
            if (u + i) % 4:  # cada usuário pula alguns filmes
                rows.append((u, i, 5.0 if scifi_fan else 1.0))
        for i in ROMANCE:
            if (u + i) % 4:
                rows.append((u, i, 1.5 if scifi_fan else 5.0))
        for i in COMEDY:
            if (u + i) % 3:
                rows.append((u, i, 3.0))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])


@pytest.fixture
def items_df() -> pd.DataFrame:
    return catalog()


@pytest.fixture
def train_df() -> pd.DataFrame:
    return ratings_df()


@pytest.fixture
def client():
    """API completa com banco SQLite populado com o catálogo de teste."""
    from app.main import app

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        for r in catalog().itertuples(index=False):
            db.add(Item(id=r.item_id, title=r.title, genres=r.genres, year=r.year, description=r.description))
        for u in range(1, 21):
            db.add(User(id=u, name=f"user{u}"))
        db.flush()
        for r in ratings_df().itertuples(index=False):
            db.add(Rating(user_id=r.user_id, item_id=r.item_id, rating=r.rating))
        db.commit()

    with TestClient(app) as c:  # dispara o lifespan (treina o modelo)
        yield c
