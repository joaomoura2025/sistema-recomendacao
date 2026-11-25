"""Filtragem baseada em conteúdo.

Cada filme vira um vetor com três blocos de features, concatenados e
normalizados (norma L2 = 1):

    [ gêneros (one-hot) | tags/descrição (TF-IDF) | década (one-hot) ]

O perfil de um usuário é a média dos vetores dos filmes que ele avaliou,
ponderada pelo quanto a nota ficou acima/abaixo da média dele. Filmes que
ele odiou "puxam o perfil para longe". Gêneros favoritos declarados no
cadastro entram como um bônus no perfil (resolve o cold start).

O score de um filme é a similaridade de cosseno entre o perfil e o vetor do
filme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

# Peso de cada bloco de features no vetor final
GENRE_WEIGHT = 1.0
TEXT_WEIGHT = 0.6
DECADE_WEIGHT = 0.3
# Peso de um gênero favorito declarado, em "avaliações equivalentes"
FAVORITE_GENRE_WEIGHT = 1.5


def split_genres(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [g for g in value if g]
    if not isinstance(value, str) or not value:
        return []
    return [g for g in value.split("|") if g and g != "(no genres listed)"]


class ContentModel:
    def __init__(self) -> None:
        self.item_ids: np.ndarray = np.array([], dtype=np.int64)
        self.index: dict[int, int] = {}
        self.matrix: sparse.csr_matrix | None = None
        self.genres: list[list[str]] = []
        self.genre_names: list[str] = []
        self._genre_offset = 0  # coluna onde começam os gêneros (sempre 0)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def fit(self, items: pd.DataFrame) -> "ContentModel":
        """items: colunas item_id, genres, year, description."""
        items = items.reset_index(drop=True)
        self.item_ids = items["item_id"].to_numpy(dtype=np.int64)
        self.index = {int(i): k for k, i in enumerate(self.item_ids)}
        self.genres = [split_genres(g) for g in items["genres"]]

        blocks = []

        mlb = MultiLabelBinarizer(sparse_output=True)
        g = mlb.fit_transform(self.genres).astype(np.float32)
        self.genre_names = list(mlb.classes_)
        blocks.append(GENRE_WEIGHT * g)

        text = items["description"].fillna("").astype(str)
        if text.str.strip().any():
            tfidf = TfidfVectorizer(
                min_df=2, max_features=20000, sublinear_tf=True, ngram_range=(1, 2),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
            )
            try:
                t = tfidf.fit_transform(text)
                blocks.append(TEXT_WEIGHT * t.astype(np.float32))
            except ValueError:  # vocabulário vazio (ex.: poucos itens com texto)
                pass

        decades = items["year"].apply(
            lambda y: [f"d{int(y) // 10 * 10}"] if pd.notna(y) else []
        )
        d = MultiLabelBinarizer(sparse_output=True).fit_transform(decades).astype(np.float32)
        if d.shape[1]:
            blocks.append(DECADE_WEIGHT * d)

        self.matrix = normalize(sparse.hstack(blocks).tocsr(), norm="l2")
        return self

    # ------------------------------------------------------------------ perfil
    def user_profile(
        self,
        ratings: dict[int, float],
        user_mean: float,
        favorite_genres: list[str] | None = None,
    ) -> np.ndarray | None:
        """Vetor de perfil do usuário (ou None se não houver nada para usar)."""
        dim = self.matrix.shape[1]
        profile = np.zeros(dim, dtype=np.float64)

        idx, weights = [], []
        for item_id, r in ratings.items():
            k = self.index.get(int(item_id))
            if k is not None:
                idx.append(k)
                # +0.5 de folga: uma nota igual à média ainda conta um pouco
                # como "gostei" (o usuário escolheu assistir aquele filme).
                weights.append(r - user_mean + 0.5)
        if idx:
            w = np.asarray(weights)
            profile += np.asarray(self.matrix[idx].T @ w).ravel() / max(len(idx), 1)

        for genre in favorite_genres or []:
            if genre in self.genre_names:
                profile[self.genre_names.index(genre)] += FAVORITE_GENRE_WEIGHT * GENRE_WEIGHT / 3

        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else None

    def score(self, profile: np.ndarray | None) -> np.ndarray:
        """Similaridade de cosseno perfil x todos os itens (vetor [-1, 1])."""
        if profile is None:
            return np.zeros(self.n_items)
        return np.asarray(self.matrix @ profile).ravel()

    def top_genres(self, profile: np.ndarray | None, n: int = 3) -> list[str]:
        if profile is None:
            return []
        g = profile[: len(self.genre_names)]
        order = np.argsort(-g)
        return [self.genre_names[i] for i in order[:n] if g[i] > 0]

    # ------------------------------------------------------------ similaridade
    def similarity_rows(self, rows: list[int]) -> np.ndarray:
        """Similaridade de cosseno entre os itens `rows` e todos os itens."""
        return np.asarray((self.matrix[rows] @ self.matrix.T).todense())

    def similar(self, item_id: int, n: int = 10) -> list[tuple[int, float]]:
        k = self.index.get(int(item_id))
        if k is None:
            return []
        sims = self.similarity_rows([k])[0]
        sims[k] = -np.inf
        top = np.argsort(-sims)[:n]
        return [(int(self.item_ids[i]), float(sims[i])) for i in top]
