"""Configurações da aplicação, lidas de variáveis de ambiente.

Todas têm valor padrão, então a aplicação roda localmente sem nenhum .env
(usando SQLite). No Docker, o docker-compose define DATABASE_URL apontando
para o PostgreSQL.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Banco de dados (SQLAlchemy URL). Padrão: SQLite local em data/app.db
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'app.db'}"
        )
    )
    # Pasta onde ficam os CSVs do MovieLens
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", BASE_DIR / "data" / "ml-latest-small"))
    )

    # --- Hiperparâmetros do modelo ---
    # Filtragem colaborativa (item-kNN)
    knn_neighbors: int = field(default_factory=lambda: _env_int("KNN_NEIGHBORS", 50))
    knn_shrinkage: float = field(default_factory=lambda: _env_float("KNN_SHRINKAGE", 10.0))
    min_item_ratings: int = field(default_factory=lambda: _env_int("MIN_ITEM_RATINGS", 5))
    # Híbrido: peso da colaborativa = n / (n + HYBRID_K), n = nº de avaliações do usuário
    hybrid_k: float = field(default_factory=lambda: _env_float("HYBRID_K", 10.0))
    # Peso do "prior" de popularidade no score final
    popularity_weight: float = field(default_factory=lambda: _env_float("POPULARITY_WEIGHT", 0.1))
    # Fração do componente de conteúdo que vem da popularidade ("populares no seu gosto")
    content_popularity_mix: float = field(
        default_factory=lambda: _env_float("CONTENT_POPULARITY_MIX", 0.8)
    )
    # Retreina automaticamente depois de N alterações (avaliações/itens). 0 = desliga
    retrain_every: int = field(default_factory=lambda: _env_int("RETRAIN_EVERY", 200))


settings = Settings()
