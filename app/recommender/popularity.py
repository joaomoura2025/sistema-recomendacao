"""Popularidade com média bayesiana.

Um filme com uma única nota 5 não deve ficar acima de um filme com 300
notas de média 4.3. A média bayesiana "puxa" a média de cada filme em
direção à média global, com força C (≈ nº de votos "fictícios"):

    score = (C * média_global + soma_notas) / (C + n_notas)

Usado (1) como recomendação para usuário totalmente novo e (2) como um
pequeno "prior" no score híbrido, para desempatar a favor de filmes bons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityModel:
    def __init__(self, c: float = 20.0) -> None:
        self.c = c
        self.scores: dict[int, float] = {}
        self.counts: dict[int, int] = {}

    def fit(self, ratings: pd.DataFrame) -> "PopularityModel":
        if ratings.empty:
            self.scores, self.counts = {}, {}
            return self
        mu = ratings["rating"].mean()
        agg = ratings.groupby("item_id")["rating"].agg(["sum", "count"])
        bayes = (self.c * mu + agg["sum"]) / (self.c + agg["count"])
        # Mistura qualidade (média bayesiana) com volume (log do nº de votos)
        vol = np.log1p(agg["count"]) / np.log1p(agg["count"].max())
        q = (bayes - bayes.min()) / max(bayes.max() - bayes.min(), 1e-9)
        final = 0.7 * q + 0.3 * vol
        self.scores = {int(i): float(s) for i, s in final.items()}
        self.counts = {int(i): int(c) for i, c in agg["count"].items()}
        return self

    def vector(self, item_ids: np.ndarray) -> np.ndarray:
        """Score de popularidade em [0, 1] alinhado com item_ids."""
        return np.array([self.scores.get(int(i), 0.0) for i in item_ids])
