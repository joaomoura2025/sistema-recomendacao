"""Avaliação offline do modelo (não precisa do banco nem da API).

Protocolo:
- Para cada usuário, separa 20% das avaliações como TESTE (aleatório, semente fixa).
- Treina o modelo com os outros 80%.
- Gera top-10 para cada usuário e compara com os filmes do teste que ele
  avaliou com nota >= 4 ("relevantes").
- Métricas: Precision@10, Recall@10, nDCG@10 e Cobertura do catálogo.
- Cenário extra "cold start": o usuário só tem 3 avaliações no treino.

Uso:
    python -m scripts.avaliar [--dados data/ml-latest-small] [--usuarios 300]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.recommender import HybridRecommender, Strategy
from scripts.seed import build_item_rows, load_frames

K = 10
RELEVANT = 4.0


def split(ratings: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(ratings), dtype=bool)
    for _, idx in ratings.groupby("user_id").indices.items():
        if len(idx) >= 5:
            n_test = max(1, int(round(len(idx) * frac)))
            test_mask[rng.choice(idx, n_test, replace=False)] = True
    return ratings[~test_mask], ratings[test_mask]


def ndcg(recs: list[int], relevant: set[int]) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(recs) if r in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), len(recs))))
    return dcg / ideal if ideal else 0.0


def evaluate(model, train, test, users, strategy, n_items, cold_start=None):
    by_user = train.groupby("user_id")
    rel_by_user = test[test.rating >= RELEVANT].groupby("user_id")["item_id"].apply(set)
    p, r, nd, rec_items = [], [], [], set()
    for u in users:
        relevant = rel_by_user.get(u)
        if not relevant:
            continue
        hist = by_user.get_group(u)
        if cold_start:
            hist = hist.sample(min(cold_start, len(hist)), random_state=int(u))
        ratings = dict(zip(hist.item_id, hist.rating))
        # Exclui todo o histórico de treino (mesmo o que não foi "mostrado")
        exclude = set(by_user.get_group(u).item_id)
        recs = [x.item_id for x in model.recommend(ratings, n=K, strategy=strategy, exclude=exclude)]
        hits = len(set(recs) & relevant)
        p.append(hits / K)
        r.append(hits / len(relevant))
        nd.append(ndcg(recs, relevant))
        rec_items.update(recs)
    return {
        "Precision@10": np.mean(p),
        "Recall@10": np.mean(r),
        "nDCG@10": np.mean(nd),
        "Cobertura": len(rec_items) / n_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", default=str(settings.data_dir))
    parser.add_argument("--usuarios", type=int, default=0, help="amostra de usuários (0 = todos)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    movies, ratings, tags = load_frames(Path(args.dados))
    items = pd.DataFrame(build_item_rows(movies, tags)).rename(columns={"id": "item_id"})
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})[
        ["user_id", "item_id", "rating"]
    ]
    train, test = split(ratings, 0.2, args.seed)

    t0 = time.time()
    model = HybridRecommender(
        knn_neighbors=settings.knn_neighbors,
        knn_shrinkage=settings.knn_shrinkage,
        min_item_ratings=settings.min_item_ratings,
        hybrid_k=settings.hybrid_k,
        popularity_weight=settings.popularity_weight,
        content_popularity_mix=settings.content_popularity_mix,
    ).fit(items, train)
    print(f"Treino: {time.time() - t0:.1f}s | {len(items)} itens, {len(train)} avaliações de treino\n")

    users = test.user_id.unique()
    if args.usuarios:
        users = np.random.default_rng(args.seed).choice(users, min(args.usuarios, len(users)), replace=False)

    rows = []
    for scenario, cold in (("Normal", None), ("Cold start (3 notas)", 3)):
        for strat in (Strategy.POPULARITY, Strategy.CONTENT, Strategy.COLLABORATIVE, Strategy.HYBRID):
            res = evaluate(model, train, test, users, strat, len(items), cold)
            rows.append({"Cenário": scenario, "Estratégia": strat.value, **res})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    try:
        print("\n" + df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
