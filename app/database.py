"""Conexão com o banco de dados (SQLAlchemy 2)."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def make_engine(url: str):
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite + FastAPI: a mesma conexão pode ser usada por threads diferentes
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


engine = make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """Dependência do FastAPI: abre uma sessão por requisição."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
