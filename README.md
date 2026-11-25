# Sistema de Recomendação de Filmes

API de recomendação de filmes com **FastAPI**, **PostgreSQL** e **Docker**,
treinada no dataset **MovieLens** (610 usuários, 9.742 filmes, 100.836
avaliações).

O modelo é **híbrido**: junta filtragem colaborativa item-item, filtragem
baseada em conteúdo e popularidade, com um peso que se adapta ao histórico de
cada usuário. Toda recomendação vem com o **motivo** em português.

```json
{
  "item": { "id": 1270, "titulo": "Back to the Future", "generos": ["Adventure", "Comedy", "Sci-Fi"], "ano": 1985 },
  "score": 0.9123,
  "componentes": { "colaborativa": 0.9379, "conteudo": 0.4572, "popularidade": 0.8168 },
  "motivo": "Quem gostou de \"Star Wars: Episode V - The Empire Strikes Back\" também gostou deste"
}
```

> A explicação detalhada do modelo, as decisões de projeto e os resultados da
> avaliação estão em **[docs/MODELO.md](docs/MODELO.md)**.

## Destaques

- **Híbrido adaptativo:** α = n/(n+10). Usuário novo recebe recomendação por
  gêneros e popularidade; quem tem muito histórico recebe quase só
  colaborativa.
- **Efeito imediato:** uma avaliação nova muda as recomendações na hora, sem
  retreinar.
- **Cold start resolvido:** gêneros favoritos no cadastro; item novo é
  recomendável assim que é cadastrado.
- **Explicações**, decomposição do score e **diversificação (MMR)** opcional.
- **Avaliação offline** com Precision@10, Recall@10, nDCG@10 e cobertura.
  O híbrido fica 50% acima da popularidade quando o usuário tem histórico e
  quase dobra a colaborativa pura no cold start.
- **52 testes** (unitários + integração).

## Como rodar

### Opção 1: Docker (recomendado)

Pré-requisitos: Docker e Docker Compose.

```bash
git clone <url-do-repositorio>
cd sistema-recomendacao
docker compose up --build
```

Na primeira vez o container baixa o MovieLens (~1 MB) e carrega no
PostgreSQL, o que leva alguns segundos. Depois disso:

- **Swagger (documentação interativa):** http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Saúde: http://localhost:8000/saude

Outros comandos:

```bash
docker compose run --rm --no-deps --entrypoint pytest api   # testes dentro do container
docker compose down        # para tudo
docker compose down -v     # para e apaga o banco (recarrega do zero no próximo up)
```

Quem tiver `make` pode usar `make up`, `make test-docker`, `make down` e
`make reset`.

### Opção 2: Local, sem Docker (SQLite)

Pré-requisito: Python 3.11 ou mais novo.

```bash
# Linux / macOS
./scripts/instalar.sh      # venv + dependências + download + carga no banco
./scripts/executar.sh      # sobe a API em http://localhost:8000/docs

# Windows (PowerShell)
.\scripts\instalar.ps1
.\scripts\executar.ps1
```

Ou manualmente:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m scripts.seed               # baixa o MovieLens e carrega no SQLite
uvicorn app.main:app --reload
```

**Sem internet?** `python -m scripts.gerar_dados_sinteticos` gera um dataset
no mesmo formato do MovieLens em `data/sintetico/`. Depois é só rodar
`DATA_DIR=data/sintetico python -m scripts.seed`.

## Endpoints

| Método | Rota | O que faz |
|---|---|---|
| `POST` | `/usuarios` | Cria usuário (opcional: `generos_favoritos`) |
| `GET` | `/usuarios/{id}` | Dados do usuário |
| `PATCH` | `/usuarios/{id}/preferencias` | Atualiza nome e gêneros favoritos |
| `GET` | `/usuarios/{id}/avaliacoes` | Lista as avaliações |
| `PUT` | `/usuarios/{id}/avaliacoes/{item_id}` | Cria ou atualiza uma nota (0,5 a 5) |
| `POST` | `/usuarios/{id}/avaliacoes` | Várias notas de uma vez |
| `DELETE` | `/usuarios/{id}/avaliacoes/{item_id}` | Remove uma nota |
| `GET` | `/usuarios/{id}/recomendacoes` | **Recomendações**. Parâmetros: `n`, `estrategia`, `genero`, `diversidade` |
| `POST` | `/itens` | Cadastra item novo |
| `GET` | `/itens` | Lista e busca itens (`busca`, `genero`, `limite`, `offset`) |
| `GET` | `/itens/{id}` | Detalhes, média e número de notas |
| `GET` | `/itens/{id}/similares` | "Quem gostou deste também gostou de…" |
| `GET` | `/generos` | Gêneros disponíveis |
| `GET` | `/modelo/status` | Quando o modelo foi treinado e com quais parâmetros |
| `POST` | `/modelo/retreinar` | Retreina o modelo com os dados atuais |
| `GET` | `/saude` | Healthcheck |

Os requisitos do enunciado ficam assim: **adicionar usuários e itens**
(`POST /usuarios`, `POST /itens`), **obter recomendações**
(`GET /usuarios/{id}/recomendacoes`) e **atualizar preferências** (endpoints
de avaliações + `PATCH /usuarios/{id}/preferencias`).

### Exemplo de uso (curl)

```bash
# 1. Cria um usuário que curte ficção científica
curl -X POST localhost:8000/usuarios -H "Content-Type: application/json" \
     -d '{"nome": "João", "generos_favoritos": ["Sci-Fi", "Thriller"]}'
# -> {"id": 611, ...}

# 2. Já dá para pedir recomendações (cold start: populares dentro dos gêneros)
curl "localhost:8000/usuarios/611/recomendacoes?n=5"

# 3. Procura um filme e avalia
curl "localhost:8000/itens?busca=matrix"
curl -X PUT localhost:8000/usuarios/611/avaliacoes/2571 \
     -H "Content-Type: application/json" -d '{"nota": 5}'

# 4. Várias notas de uma vez
curl -X POST localhost:8000/usuarios/611/avaliacoes -H "Content-Type: application/json" \
     -d '{"avaliacoes": [{"item_id": 1196, "nota": 5}, {"item_id": 4993, "nota": 2}]}'

# 5. Recomendações atualizadas, mais diversas, só comédias
curl "localhost:8000/usuarios/611/recomendacoes?n=10&diversidade=0.5"
curl "localhost:8000/usuarios/611/recomendacoes?genero=Comedy"

# 6. Compara os algoritmos
curl "localhost:8000/usuarios/611/recomendacoes?estrategia=colaborativa"
curl "localhost:8000/usuarios/611/recomendacoes?estrategia=conteudo"
```

## Testes e avaliação

```bash
python -m pytest                  # 52 testes, ~2 segundos
python -m pytest -m integracao    # só os de integração (API + banco)
python -m scripts.avaliar         # avaliação offline no MovieLens (~15 s)
```

- `tests/test_modelo.py`: **testes unitários** do núcleo (conteúdo,
  colaborativa, popularidade, híbrido), num catálogo pequeno em que se sabe de
  antemão o que deve ser recomendado.
- `tests/test_api.py`: **testes de integração**, que sobem a API completa com
  banco (validações, 404/422, cold start, efeito imediato das notas, item novo,
  retreino manual e automático).

Resultado da avaliação (todos os 610 usuários; detalhes em
[docs/MODELO.md](docs/MODELO.md#7-avaliação-offline)):

| Cenário | Popularidade | Conteúdo | Colaborativa | **Híbrido** |
|---|--:|--:|--:|--:|
| Normal: Precision@10 | 0,116 | 0,005 | 0,173 | **0,173** |
| Cold start (3 notas): Precision@10 | 0,116 | 0,006 | 0,055 | **0,102** |

## Configuração

Tudo por variável de ambiente (todas opcionais):

| Variável | Padrão | Descrição |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/app.db` | URL do SQLAlchemy (no Docker: PostgreSQL) |
| `DATA_DIR` | `data/ml-latest-small` | Pasta dos CSVs do MovieLens |
| `SEED_ON_STARTUP` | `true` | Carrega os dados ao subir o container |
| `RETRAIN_EVERY` | `200` | Retreina sozinho após N alterações (0 = desliga) |
| `KNN_NEIGHBORS` | `50` | Vizinhos por filme na colaborativa |
| `KNN_SHRINKAGE` | `10` | λ do shrinkage |
| `MIN_ITEM_RATINGS` | `5` | Mínimo de notas para o filme entrar na colaborativa |
| `HYBRID_K` | `10` | K de α = n/(n+K) |
| `POPULARITY_WEIGHT` | `0.1` | Peso do prior de popularidade |
| `CONTENT_POPULARITY_MIX` | `0.8` | Fração de popularidade dentro do componente de conteúdo |

## Estrutura do projeto

```
├── app/
│   ├── main.py                  # app FastAPI, lifespan, /saude, /generos
│   ├── config.py                # configurações (variáveis de ambiente)
│   ├── database.py              # engine e sessão do SQLAlchemy
│   ├── models.py                # tabelas: users, items, ratings
│   ├── schemas.py               # entrada/saída da API (Pydantic)
│   ├── routers/                 # endpoints: usuarios, itens, modelo
│   ├── services/
│   │   └── recommender_service.py   # modelo em memória, treino, retreino
│   └── recommender/             # núcleo do modelo (sem banco, sem FastAPI)
│       ├── collaborative.py     # item-kNN com shrinkage
│       ├── content.py           # TF-IDF + gêneros + década
│       ├── popularity.py        # média bayesiana
│       └── hybrid.py            # combinação, MMR, explicações
├── scripts/
│   ├── download_movielens.py    # baixa o dataset
│   ├── seed.py                  # carrega no banco (idempotente)
│   ├── avaliar.py               # avaliação offline
│   ├── gerar_dados_sinteticos.py
│   ├── entrypoint.sh            # inicialização do container
│   └── instalar/executar (.sh e .ps1)
├── tests/                       # pytest: unitários + integração
├── docs/MODELO.md               # modelo, decisões e resultados
├── Dockerfile
├── docker-compose.yml           # API + PostgreSQL 16
└── requirements*.txt
```

## Dados

MovieLens latest-small, da GroupLens Research
(F. Maxwell Harper e Joseph A. Konstan, 2015, *The MovieLens Datasets:
History and Context*, ACM TiiS 5(4)). O dataset não é versionado no
repositório: ele é baixado automaticamente da GroupLens, com uma cópia no
GitHub como plano B.

## Integrantes

- João Vitor
