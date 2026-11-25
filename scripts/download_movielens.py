"""Baixa o dataset MovieLens "latest-small" (~100 mil avaliações, 610 usuários,
~9.700 filmes) da GroupLens e descompacta em data/ml-latest-small/.

Uso:
    python -m scripts.download_movielens
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from app.config import settings

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# Plano B, se o site da GroupLens estiver fora do ar: cópia idêntica dos CSVs
# num repositório público do GitHub (mesmas 100.836 avaliações).
MIRROR = "https://raw.githubusercontent.com/speegled/fosdata/master/data-raw/ml-latest-small/{}"
REQUIRED = ("movies.csv", "ratings.csv", "tags.csv")


def is_downloaded(data_dir: Path = settings.data_dir) -> bool:
    return all((data_dir / name).exists() for name in REQUIRED)


def download(data_dir: Path = settings.data_dir) -> Path:
    if is_downloaded(data_dir):
        print(f"[download] Dataset já existe em {data_dir}")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[download] Baixando {URL} ...")
        with urllib.request.urlopen(URL, timeout=120) as resp:
            payload = resp.read()
        # O zip tem uma pasta "ml-latest-small/" dentro; extraímos os CSVs
        # direto em data_dir, seja qual for o nome dela.
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name.endswith(".csv") or name == "README.txt":
                    (data_dir / name).write_bytes(zf.read(member))
    except Exception as exc:  # noqa: BLE001
        print(f"[download] GroupLens indisponível ({exc}); tentando o espelho no GitHub...")
        for name in REQUIRED:
            with urllib.request.urlopen(MIRROR.format(name), timeout=120) as resp:
                (data_dir / name).write_bytes(resp.read())

    print(f"[download] OK -> {data_dir}")
    return data_dir


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:  # noqa: BLE001
        print(f"[download] Falhou: {exc}", file=sys.stderr)
        print(
            "[download] Sem internet? Gere dados sintéticos com:\n"
            "    python -m scripts.gerar_dados_sinteticos",
            file=sys.stderr,
        )
        sys.exit(1)
