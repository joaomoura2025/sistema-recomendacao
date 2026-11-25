"""Núcleo do sistema de recomendação.

Não depende de banco nem de FastAPI: recebe DataFrames do pandas e devolve
recomendações. Isso deixa o modelo fácil de testar e de avaliar offline
(scripts/avaliar.py).
"""

from app.recommender.collaborative import ItemKNN
from app.recommender.content import ContentModel
from app.recommender.hybrid import HybridRecommender, Recommendation, Strategy
from app.recommender.popularity import PopularityModel

__all__ = [
    "ContentModel",
    "HybridRecommender",
    "ItemKNN",
    "PopularityModel",
    "Recommendation",
    "Strategy",
]
