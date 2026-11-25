"""Cria as tabelas e carrega o MovieLens no banco.

- Se o banco já tiver filmes, não faz nada (é seguro rodar várias vezes).
- Se os CSVs não existirem, tenta baixar.

Uso:
    python -m scripts.seed            # carrega se estiver vazio
    python -m scripts.seed --force    # apaga tudo e recarrega
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import func, insert, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import Item, Rating, User
from scripts.download_movielens import download, is_downloaded

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
BATCH = 5000


def parse_year(title: str) -> int | None:
    m = YEAR_RE.search(title.strip())
    return int(m.group(1)) if m else None


def clean_title(title: str) -> str:
    return YEAR_RE.sub("", title.strip()).strip()


def load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    tags_path = data_dir / "tags.csv"
    tags = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame(
        columns=["userId", "movieId", "tag", "timestamp"]
    )
    return movies, ratings, tags


def build_item_rows(movies: pd.DataFrame, tags: pd.DataFrame) -> list[dict]:
    # Junta todas as tags de um filme num texto só (vira "descrição")
    tags = tags.dropna(subset=["tag"])
    tag_text = (
        tags.assign(tag=tags["tag"].astype(str).str.lower().str.strip())
        .groupby("movieId")["tag"]
        .apply(lambda s: " ".join(sorted(set(s))))
    )
    rows = []
    for m in movies.itertuples(index=False):
        genres = "" if m.genres == "(no genres listed)" else m.genres
        rows.append(
            {
                "id": int(m.movieId),
                "title": clean_title(m.title),
                "year": parse_year(m.title),
                "genres": genres,
                "description": tag_text.get(m.movieId, ""),
            }
        )
    return rows


def bulk_insert(db: Session, model, rows: list[dict]) -> None:
    for start in range(0, len(rows), BATCH):
        db.execute(insert(model), rows[start : start + BATCH])


def fix_sequences(db: Session) -> None:
    """No PostgreSQL, inserir IDs explícitos não avança a sequence do SERIAL.
    Sem isso, o primeiro POST /usuarios daria conflito de chave primária."""
    if db.bind.dialect.name != "postgresql":
        return
    for table in ("users", "items", "ratings"):
        db.execute(
            text(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
            )
        )


def seed(force: bool = False, data_dir: Path = settings.data_dir) -> None:
    Base.metadata.create_all(engine)

    with SessionLocal() as db:
        n_items = db.scalar(select(func.count()).select_from(Item))
        if n_items and not force:
            print(f"[seed] Banco já tem {n_items} itens. Nada a fazer (use --force para recarregar).")
            return

        if not is_downloaded(data_dir):
            download(data_dir)

        movies, ratings, tags = load_frames(data_dir)
        print(f"[seed] {len(movies)} filmes, {ratings.userId.nunique()} usuários, {len(ratings)} avaliações")

        if force:
            for model in (Rating, Item, User):
                db.query(model).delete()

        bulk_insert(db, Item, build_item_rows(movies, tags))
        bulk_insert(
            db,
            User,
            [
                {"id": int(u), "name": f"MovieLens #{u}", "favorite_genres": ""}
                for u in sorted(ratings.userId.unique())
            ],
        )
        bulk_insert(
            db,
            Rating,
            [
                {
                    "user_id": int(r.userId),
                    "item_id": int(r.movieId),
                    "rating": float(r.rating),
                    "timestamp": datetime.fromtimestamp(int(r.timestamp), tz=timezone.utc),
                }
                for r in ratings.itertuples(index=False)
            ],
        )
        fix_sequences(db)
        db.commit()
        print("[seed] Carga concluída.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="apaga e recarrega os dados")
    args = parser.parse_args()
    try:
        seed(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"[seed] Erro: {exc}", file=sys.stderr)
        sys.exit(1)
