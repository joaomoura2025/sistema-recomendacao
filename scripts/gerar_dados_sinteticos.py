"""Gera um dataset sintético no MESMO formato do MovieLens (movies.csv,
ratings.csv, tags.csv). Serve para rodar o projeto sem internet e para
testes. Os dados têm estrutura latente (usuários gostam de certos gêneros),
então o modelo tem algo real para aprender.

Uso:
    python -m scripts.gerar_dados_sinteticos [--saida data/sintetico]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
TAG_WORDS = {
    "Action": ["explosions", "car chase", "fight scenes"],
    "Sci-Fi": ["space", "aliens", "dystopia", "time travel"],
    "Romance": ["love story", "romantic", "wedding"],
    "Horror": ["scary", "gore", "zombies"],
    "Comedy": ["funny", "satire", "slapstick"],
    "Animation": ["pixar", "cartoon"],
    "Crime": ["heist", "mafia", "detective"],
    "Drama": ["emotional", "based on a true story"],
}


def generate(out: Path, n_users: int = 600, n_movies: int = 3000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    g = len(GENRES)

    # Filmes: 1 a 3 gêneros, ano, "qualidade" latente e popularidade (lei de potência)
    movie_genres = np.zeros((n_movies, g))
    titles, genre_str, tags_rows = [], [], []
    for m in range(n_movies):
        k = rng.integers(1, 4)
        chosen = rng.choice(g, size=k, replace=False)
        movie_genres[m, chosen] = 1
        year = int(rng.integers(1950, 2019))
        titles.append(f"Filme Sintetico {m + 1} ({year})")
        genre_str.append("|".join(GENRES[c] for c in chosen))
        for c in chosen:
            words = TAG_WORDS.get(GENRES[c])
            if words and rng.random() < 0.5:
                tags_rows.append((1, m + 1, rng.choice(words), 1500000000))
    quality = rng.normal(0, 0.5, n_movies)
    popularity = rng.pareto(1.2, n_movies) + 0.05
    popularity /= popularity.sum()

    # Usuários: afinidade por gênero (poucos gêneros muito queridos)
    affinity = rng.normal(0, 0.6, (n_users, g))
    for u in range(n_users):
        affinity[u, rng.choice(g, size=3, replace=False)] += 1.5
    bias = rng.normal(0, 0.4, n_users)

    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.lognormal(3.8, 0.8), 20, 800))
        # Usuários tendem a assistir filmes dos gêneros de que gostam
        pref = movie_genres @ np.maximum(affinity[u], 0) + 0.2
        p = popularity * pref
        p /= p.sum()
        seen = rng.choice(n_movies, size=min(n, n_movies), replace=False, p=p)
        for m in seen:
            score = (
                3.4 + bias[u] + quality[m]
                + 0.6 * (movie_genres[m] @ affinity[u]) / movie_genres[m].sum()
                + rng.normal(0, 0.6)
            )
            rating = float(np.clip(np.round(score * 2) / 2, 0.5, 5.0))
            rows.append((u + 1, m + 1, rating, 1500000000 + int(rng.integers(0, 10**8))))

    pd.DataFrame({"movieId": range(1, n_movies + 1), "title": titles, "genres": genre_str}).to_csv(
        out / "movies.csv", index=False
    )
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(
        out / "ratings.csv", index=False
    )
    pd.DataFrame(tags_rows, columns=["userId", "movieId", "tag", "timestamp"]).to_csv(
        out / "tags.csv", index=False
    )
    print(f"[sintetico] {n_movies} filmes, {n_users} usuários, {len(rows)} avaliações -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", default="data/sintetico")
    args = parser.parse_args()
    generate(Path(args.saida))
