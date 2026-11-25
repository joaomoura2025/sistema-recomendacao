"""Recomendador híbrido: colaborativa + conteúdo + popularidade.

score_final = α · colaborativa + (1 − α) · conteúdo + w_pop · popularidade

O peso α NÃO é fixo: depende de quantas avaliações o usuário tem.

    α = n / (n + K)        (K = 10 por padrão)

- Usuário com 0 avaliações  -> α = 0    (só conteúdo/gêneros favoritos)
- Usuário com 10 avaliações -> α = 0.5
- Usuário com 100           -> α ≈ 0.91 (quase só colaborativa)

Faz sentido porque a colaborativa precisa de histórico para funcionar e a
de conteúdo não. Depois do ranking ainda há uma etapa opcional de
diversificação (MMR) e cada recomendação vem com o motivo em português.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from app.recommender.collaborative import ItemKNN
from app.recommender.content import ContentModel, split_genres
from app.recommender.popularity import PopularityModel


class Strategy(str, Enum):
    HYBRID = "hibrido"
    COLLABORATIVE = "colaborativa"
    CONTENT = "conteudo"
    POPULARITY = "popularidade"


@dataclass
class Recommendation:
    item_id: int
    score: float
    collaborative: float
    content: float
    popularity: float
    reason: str


@dataclass
class HybridRecommender:
    knn_neighbors: int = 50
    knn_shrinkage: float = 10.0
    min_item_ratings: int = 5
    hybrid_k: float = 10.0
    popularity_weight: float = 0.1
    # Fração do componente "conteúdo" que vem da popularidade (ver docstring)
    content_popularity_mix: float = 0.8

    content: ContentModel = field(default_factory=ContentModel)
    cf: ItemKNN = field(init=False)
    popularity: PopularityModel = field(default_factory=PopularityModel)
    titles: dict[int, str] = field(default_factory=dict)
    global_mean: float = 3.5
    _cf_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def __post_init__(self) -> None:
        self.cf = ItemKNN(self.knn_neighbors, self.knn_shrinkage, self.min_item_ratings)

    # ------------------------------------------------------------------ treino
    def fit(self, items: pd.DataFrame, ratings: pd.DataFrame) -> "HybridRecommender":
        """items: item_id, title, genres, year, description
        ratings: user_id, item_id, rating"""
        self.fit_content(items)
        self.cf.fit(ratings)
        self.popularity.fit(ratings)
        if not ratings.empty:
            self.global_mean = float(ratings["rating"].mean())
        self._align()
        return self

    def fit_content(self, items: pd.DataFrame) -> None:
        """Retreina só a parte de conteúdo (rápido). Chamado quando entra um
        item novo: ele passa a ser recomendável na hora, mesmo sem notas."""
        self.content.fit(items)
        self.titles = dict(zip(items["item_id"].astype(int), items["title"]))
        self._align()

    def _align(self) -> None:
        """Mapeia a posição de cada item do modelo de conteúdo (que tem TODOS
        os itens) para a posição na colaborativa (que só tem itens com
        avaliações suficientes); -1 quando o item não está na colaborativa."""
        self._cf_pos = np.array(
            [self.cf.index.get(int(i), -1) for i in self.content.item_ids], dtype=np.int64
        )

    @property
    def is_ready(self) -> bool:
        return self.content.n_items > 0

    # ------------------------------------------------------------ recomendação
    def alpha(self, ratings: dict[int, float]) -> float:
        n = sum(1 for i in ratings if int(i) in self.cf.index)
        return n / (n + self.hybrid_k) if n else 0.0

    def recommend(
        self,
        ratings: dict[int, float],
        favorite_genres: list[str] | None = None,
        n: int = 10,
        strategy: Strategy = Strategy.HYBRID,
        genre: str | None = None,
        diversity: float = 0.0,
        exclude: set[int] | None = None,
    ) -> list[Recommendation]:
        if not self.is_ready:
            return []

        ratings = {int(k): float(v) for k, v in ratings.items()}
        user_mean = float(np.mean(list(ratings.values()))) if ratings else self.global_mean

        # --- componentes, todos alinhados com self.content.item_ids
        cf_raw = np.zeros(self.content.n_items)
        if strategy in (Strategy.HYBRID, Strategy.COLLABORATIVE):
            s = self.cf.score(ratings, user_mean)
            mask = self._cf_pos >= 0
            cf_raw[mask] = s[self._cf_pos[mask]] if len(s) else 0.0

        profile = None
        ct_raw = np.zeros(self.content.n_items)
        if strategy in (Strategy.HYBRID, Strategy.CONTENT):
            profile = self.content.user_profile(ratings, user_mean, favorite_genres)
            ct_raw = self.content.score(profile)

        pop = self.popularity.vector(self.content.item_ids)

        cf = _scale(cf_raw)
        ct = _scale(ct_raw)

        if strategy == Strategy.COLLABORATIVE:
            a, w_pop = 1.0, 0.0
        elif strategy == Strategy.CONTENT:
            a, w_pop = 0.0, 0.0
        elif strategy == Strategy.POPULARITY:
            a, w_pop = 0.0, 1.0
        else:
            a, w_pop = self.alpha(ratings), self.popularity_weight

        no_signal = not cf_raw.any() and not ct_raw.any()
        if no_signal:
            # Usuário sem histórico e sem gêneros favoritos: só popularidade
            w_pop = 1.0
        if strategy == Strategy.HYBRID:
            # Conteúdo puro tende a sugerir filmes obscuros que só "batem" no
            # gênero. Misturar com popularidade = "populares dentro do seu gosto".
            m = self.content_popularity_mix
            ct_mix = (1 - m) * ct + m * pop
        else:
            ct_mix = ct
        final = a * cf + (1 - a) * ct_mix + w_pop * pop

        # --- filtros: já avaliados, excluídos e gênero
        blocked = set(ratings) | (exclude or set())
        valid = np.ones(self.content.n_items, dtype=bool)
        for item_id in blocked:
            k = self.content.index.get(item_id)
            if k is not None:
                valid[k] = False
        if genre:
            valid &= np.array([genre in g for g in self.content.genres])
        if not ratings and ct_raw.any():
            # Só gêneros declarados, nenhuma nota: recomenda os populares
            # DENTRO desses gêneros (sem isso, um drama muito popular
            # apareceria para quem disse que só curte ficção científica).
            valid &= ct_raw > 0
        if strategy == Strategy.COLLABORATIVE and not no_signal:
            valid &= cf_raw > 0
        final = np.where(valid, final, -np.inf)

        n_valid = int(np.isfinite(final).sum())
        if n_valid == 0:
            return []
        n = min(n, n_valid)

        if diversity > 0:
            chosen = self._mmr(final, n, diversity)
        else:
            chosen = np.argsort(-final)[:n].tolist()

        top_genres = self.content.top_genres(profile, 4) if profile is not None else []
        out = []
        for k in chosen:
            item_id = int(self.content.item_ids[k])
            out.append(
                Recommendation(
                    item_id=item_id,
                    score=round(float(final[k]), 4),
                    collaborative=round(float(cf[k]), 4),
                    content=round(float(ct[k]), 4),
                    popularity=round(float(pop[k]), 4),
                    reason=self._explain(
                        k, item_id, a * cf[k], (1 - a) * ct_mix[k], ratings, user_mean,
                        top_genres, no_signal or strategy == Strategy.POPULARITY,
                    ),
                )
            )
        return out

    def _mmr(self, scores: np.ndarray, n: int, lam: float) -> list[int]:
        """Maximal Marginal Relevance: escolhe itens um a um equilibrando
        relevância e "não ser parecido com o que já foi escolhido".
        lam = 0 -> só relevância; lam = 1 -> só diversidade."""
        pool = min(max(n * 5, 50), int(np.isfinite(scores).sum()))
        cand = np.argsort(-scores)[:pool]
        rel = scores[cand]
        rel = (rel - rel.min()) / max(rel.max() - rel.min(), 1e-9)
        sim = self.content.similarity_rows(cand.tolist())[:, cand]

        selected: list[int] = []
        max_sim = np.zeros(len(cand))
        available = np.ones(len(cand), dtype=bool)
        for _ in range(n):
            mmr = (1 - lam) * rel - lam * max_sim
            mmr[~available] = -np.inf
            best = int(np.argmax(mmr))
            selected.append(best)
            available[best] = False
            max_sim = np.maximum(max_sim, sim[best])
        return [int(cand[i]) for i in selected]

    def _explain(
        self, k, item_id, cf_part, ct_part, ratings, user_mean, top_genres, popular_only
    ) -> str:
        if popular_only:
            return "Popular e bem avaliado pelos usuários"
        if cf_part >= ct_part and cf_part > 0:
            src = self.cf.top_contributor(item_id, ratings, user_mean)
            if src is not None:
                return f'Quem gostou de "{self.titles.get(src, src)}" também gostou deste'
        common = [g for g in top_genres if g in self.content.genres[k]]
        if common:
            return "Combina com gêneros que você curte: " + ", ".join(common)
        return "Parecido com filmes que você avaliou bem"

    # ------------------------------------------------------------- similares
    def similar_items(self, item_id: int, n: int = 10) -> list[tuple[int, float, str]]:
        """Itens parecidos: usa a colaborativa se o item tiver vizinhos,
        senão cai para similaridade de conteúdo."""
        cf = self.cf.similar(item_id, n)
        if len(cf) >= n:
            return [(i, s, "colaborativa") for i, s in cf]
        seen = {i for i, _ in cf} | {int(item_id)}
        extra = [(i, s, "conteudo") for i, s in self.content.similar(item_id, n * 2) if i not in seen]
        return [(i, s, "colaborativa") for i, s in cf] + extra[: n - len(cf)]

    def known_genres(self) -> list[str]:
        return list(self.content.genre_names)


def _scale(v: np.ndarray) -> np.ndarray:
    """Divide pelo maior valor absoluto -> fica em [-1, 1], preservando o zero."""
    m = np.abs(v).max() if len(v) else 0.0
    return v / m if m > 0 else v


__all__ = ["HybridRecommender", "Recommendation", "Strategy", "split_genres"]
