"""Filtragem colaborativa item-item (item-kNN).

Ideia: dois filmes são parecidos se as MESMAS pessoas deram notas parecidas
para eles. Passos do treino:

1. Centraliza as notas de cada usuário (nota - média do usuário). Assim um
   usuário "bonzinho", que dá 4 para tudo, e um "chato", que dá 2, ficam
   comparáveis.
2. Calcula a similaridade de cosseno entre as colunas (filmes) dessa matriz.
3. Aplica "shrinkage": similaridade * n_comum / (n_comum + λ). Se só 2
   pessoas viram os dois filmes, a similaridade é pouco confiável e é
   encolhida em direção a zero.
4. Guarda só os K vizinhos mais parecidos de cada filme (matriz esparsa).

Na hora de recomendar NÃO é preciso retreinar: o score de um filme i para o
usuário u é calculado na hora, a partir das notas atuais dele:

    score(u, i) = Σ_j sim(i, j) * (r_uj - média_u + b)

(j = vizinhos de i que u avaliou; b = 0.5 é um bônus por "ter assistido":
escolher ver um filme já é um sinal de interesse, mesmo com nota na média).

Repare que NÃO dividimos por Σ sim (como na fórmula clássica de previsão de
nota). Para ranking top-N, a soma funciona bem melhor: na avaliação offline
a versão normalizada teve Precision@10 = 0.098 e a soma 0.179 (ver
docs/MODELO.md). A soma premia filmes com muitos vizinhos que o usuário
curtiu, ou seja, com mais evidência.
Como o score é calculado na hora, uma avaliação nova muda as recomendações
imediatamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

CHUNK = 1024  # nº de itens processados por vez (limita o uso de memória)


class ItemKNN:
    def __init__(
        self,
        k: int = 50,
        shrinkage: float = 10.0,
        min_item_ratings: int = 5,
        watch_bonus: float = 0.5,
    ) -> None:
        self.k = k
        self.watch_bonus = watch_bonus
        self.shrinkage = shrinkage
        self.min_item_ratings = min_item_ratings
        self.item_ids: np.ndarray = np.array([], dtype=np.int64)
        self.index: dict[int, int] = {}
        self.sim: sparse.csr_matrix | None = None  # (n_items x n_items), linha i = vizinhos de i
        self.global_mean: float = 3.5

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def fit(self, ratings: pd.DataFrame) -> "ItemKNN":
        """ratings: colunas user_id, item_id, rating."""
        if ratings.empty:
            self.sim = sparse.csr_matrix((0, 0), dtype=np.float32)
            return self

        self.global_mean = float(ratings["rating"].mean())
        user_mean = ratings.groupby("user_id")["rating"].transform("mean")
        df = ratings.assign(centered=ratings["rating"] - user_mean)

        counts = df["item_id"].value_counts()
        keep = counts[counts >= self.min_item_ratings].index
        df = df[df["item_id"].isin(keep)]

        self.item_ids = np.sort(df["item_id"].unique()).astype(np.int64)
        self.index = {int(i): k for k, i in enumerate(self.item_ids)}
        users = {u: k for k, u in enumerate(df["user_id"].unique())}
        n_items, n_users = len(self.item_ids), len(users)

        rows = df["user_id"].map(users).to_numpy()
        cols = df["item_id"].map(self.index).to_numpy()
        r = sparse.csr_matrix(
            (df["centered"].to_numpy(np.float32), (rows, cols)), shape=(n_users, n_items)
        )
        b = sparse.csr_matrix(
            (np.ones(len(df), np.float32), (rows, cols)), shape=(n_users, n_items)
        )
        rt = r.T.tocsr()
        bt = b.T.tocsr()
        norms = np.sqrt(np.asarray(r.multiply(r).sum(axis=0)).ravel()) + 1e-9

        k = min(self.k, max(n_items - 1, 0))
        data, indices, indptr = [], [], [0]
        for start in range(0, n_items, CHUNK):
            stop = min(start + CHUNK, n_items)
            dot = (rt[start:stop] @ r).toarray()           # produto interno
            common = (bt[start:stop] @ b).toarray()        # nº de usuários em comum
            sim = dot / (norms[start:stop, None] * norms[None, :])
            sim *= common / (common + self.shrinkage)      # shrinkage
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0  # sem auto-similaridade
            sim[sim < 0] = 0.0  # só vizinhos positivamente correlacionados

            if k > 0:
                top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            for row in range(stop - start):
                cols_row = top[row] if k > 0 else np.array([], dtype=int)
                vals = sim[row, cols_row]
                mask = vals > 0
                cols_row, vals = cols_row[mask], vals[mask]
                order = np.argsort(cols_row)
                indices.extend(cols_row[order].tolist())
                data.extend(vals[order].tolist())
                indptr.append(len(indices))

        self.sim = sparse.csr_matrix(
            (np.asarray(data, np.float32), np.asarray(indices, np.int32), np.asarray(indptr)),
            shape=(n_items, n_items),
        )
        return self

    def _user_vectors(self, ratings: dict[int, float], user_mean: float):
        d = np.zeros(self.n_items, dtype=np.float32)
        seen = np.zeros(self.n_items, dtype=np.float32)
        for item_id, r in ratings.items():
            k = self.index.get(int(item_id))
            if k is not None:
                d[k] = r - user_mean
                seen[k] = 1.0
        return d, seen

    def score(self, ratings: dict[int, float], user_mean: float) -> np.ndarray:
        """Score de cada item do modelo para o usuário (quanto maior, mais
        evidência de que ele vai gostar)."""
        if not self.n_items or not ratings:
            return np.zeros(self.n_items)
        d, seen = self._user_vectors(ratings, user_mean)
        return self.sim @ (d + self.watch_bonus * seen)

    def top_contributor(
        self, item_id: int, ratings: dict[int, float], user_mean: float
    ) -> int | None:
        """Qual filme avaliado pelo usuário mais "puxou" este item para cima.
        Usado para explicar a recomendação."""
        k = self.index.get(int(item_id))
        if k is None:
            return None
        row = self.sim.getrow(k)
        best, best_val = None, 0.0
        for j, s in zip(row.indices, row.data):
            other = int(self.item_ids[j])
            if other in ratings:
                contrib = s * (ratings[other] - user_mean + self.watch_bonus)
                if contrib > best_val:
                    best, best_val = other, contrib
        return best

    def similar(self, item_id: int, n: int = 10) -> list[tuple[int, float]]:
        k = self.index.get(int(item_id))
        if k is None:
            return []
        row = self.sim.getrow(k)
        order = np.argsort(-row.data)[:n]
        return [(int(self.item_ids[row.indices[i]]), float(row.data[i])) for i in order]
