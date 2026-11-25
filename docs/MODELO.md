# Modelo de recomendação e decisões de projeto

Este documento descreve o modelo usado, por que ele foi montado assim e como
ele se saiu na avaliação offline.

## 1. Visão geral

O sistema é **híbrido** e combina três sinais:

| Sinal | O que captura | Ponto forte | Ponto fraco |
|---|---|---|---|
| **Filtragem colaborativa** (item-kNN) | "Quem gostou de X também gostou de Y" | A mais precisa quando há histórico | Não funciona sem notas (usuário ou item novo) |
| **Baseada em conteúdo** (gêneros + tags + década) | Filmes parecidos em características | Funciona para item novo e para usuário que só declarou gêneros | Sozinha, sugere filmes obscuros |
| **Popularidade** (média bayesiana) | Filmes bons e muito vistos | Funciona sem dado nenhum do usuário | Igual para todo mundo, sem personalização |

```
score = α · colaborativa + (1 − α) · [ (1 − m) · conteúdo + m · popularidade ] + w · popularidade

α = n / (n + K)       n = nº de avaliações do usuário, K = 10
m = 0.8               mistura conteúdo/popularidade ("populares dentro do seu gosto")
w = 0.1               prior de popularidade (desempate a favor de filmes bons)
```

Antes de somar, a colaborativa e o conteúdo são divididos pelo maior valor
absoluto entre os candidatos, para ficarem na mesma escala ([-1, 1]).

### O peso adaptativo α

O α diz quanto a colaborativa pesa para cada usuário:

| Avaliações do usuário | α | O que domina |
|---:|---:|---|
| 0 | 0 | Gêneros favoritos + popularidade |
| 3 | 0,23 | Principalmente conteúdo/popularidade |
| 10 | 0,50 | Metade/metade |
| 50 | 0,83 | Principalmente colaborativa |
| 200 | 0,95 | Quase só colaborativa |

A API devolve esse valor (`peso_colaborativa`) em toda resposta de
recomendação, então dá para ver a transição acontecendo.

## 2. Filtragem colaborativa: item-kNN

**Treino** (`app/recommender/collaborative.py`):

1. **Centralização:** cada nota vira `nota − média do usuário`. Assim um
   usuário que dá 4 para tudo e outro que dá 2 para tudo ficam comparáveis.
2. **Similaridade de cosseno** entre as colunas (filmes) dessa matriz.
3. **Shrinkage:** `sim × n_comum / (n_comum + λ)`, com λ = 10. Se só duas
   pessoas viram os dois filmes, a similaridade é pouco confiável e é puxada
   para zero.
4. **Poda:** só similaridades positivas e só os K = 50 vizinhos mais
   parecidos de cada filme. A matriz fica esparsa e pequena.
5. Filmes com menos de 5 avaliações ficam de fora da colaborativa, mas
   continuam recomendáveis pelo conteúdo.

O cálculo é feito em blocos de 1.024 filmes, para não montar uma matriz
densa de 9.700 × 9.700 na memória.

**Recomendação:**

```
score(u, i) = Σ_j  sim(i, j) · (nota_uj − média_u + 0.5)
              j = vizinhos de i que u avaliou
```

O `+0.5` é um bônus por ter assistido: escolher ver um filme já é sinal de
interesse, mesmo quando a nota ficou na média.

> **Decisão importante: soma em vez de média ponderada.** A fórmula clássica
> divide por Σ|sim| para *prever a nota*. Mas o objetivo aqui é *ordenar* um
> top-10, e nesse caso a soma funciona muito melhor, porque premia os filmes
> com mais evidência. Na avaliação, a versão normalizada teve
> Precision@10 = 0,098 e a soma teve 0,179.

**Consequência prática:** como o score é calculado na hora, a partir das
notas atuais que estão no banco, **uma avaliação nova muda as recomendações
imediatamente**, sem retreinar. Só a matriz de similaridade fica "congelada"
até o próximo treino.

## 3. Filtragem baseada em conteúdo

Cada filme vira um vetor (`app/recommender/content.py`):

```
[ gêneros one-hot (peso 1.0) | tags/descrição TF-IDF 1-2 gramas (peso 0.6) | década one-hot (peso 0.3) ]
```

O vetor é normalizado (norma L2 = 1). No MovieLens, as tags dos usuários
(`tags.csv`) viram a "descrição" de cada filme; para itens cadastrados pela
API, entra o campo `descricao`.

**Perfil do usuário** = média dos vetores dos filmes avaliados, ponderada por
`nota − média + 0.5`. Filmes de que ele não gostou puxam o perfil para longe.
Os gêneros favoritos declarados no cadastro somam um bônus ao perfil. O score
de cada filme é o cosseno entre o perfil e o vetor do filme.

## 4. Popularidade: média bayesiana

```
média_bayes = (C · média_global + soma_das_notas) / (C + nº_de_notas),   C = 20
popularidade = 0.7 · média_bayes (normalizada) + 0.3 · log(1 + nº_de_notas) (normalizado)
```

Assim, um filme com uma única nota 5 não passa na frente de um filme com 300
notas e média 4,3.

## 5. Cold start (usuário ou item novo)

| Situação | O que acontece |
|---|---|
| Usuário novo **sem** gêneros favoritos | Top populares (`"Popular e bem avaliado pelos usuários"`) |
| Usuário novo **com** gêneros favoritos | Só filmes desses gêneros, ordenados por popularidade |
| Usuário com poucas notas | α pequeno: conteúdo + popularidade dominam |
| Item novo (`POST /itens`) | O modelo de conteúdo é atualizado na hora (< 1 s); o item já pode ser recomendado e tem "similares" |
| Item novo recebe notas | Entra na colaborativa no próximo retreino |

## 6. Recursos extras (originalidade)

- **Explicação de cada recomendação**, em português:
  - colaborativa: `Quem gostou de "Matrix, The" também gostou deste`, citando
    o filme avaliado que mais contribuiu para o score;
  - conteúdo: `Combina com gêneros que você curte: Sci-Fi, Thriller`;
  - popularidade: `Popular e bem avaliado pelos usuários`.
- **Diversificação (MMR):** com o parâmetro `?diversidade=0.5`, a lista é
  montada item a item, equilibrando relevância e "não ser parecido com o que
  já entrou". Evita, por exemplo, um top-10 só com filmes da mesma franquia.
- **Decomposição do score:** cada recomendação mostra quanto veio da
  colaborativa, do conteúdo e da popularidade.
- **Estratégia escolhida na requisição** (`?estrategia=colaborativa|conteudo|popularidade|hibrido`),
  útil para comparar os algoritmos no próprio Swagger.
- **Retreino automático em segundo plano** a cada N alterações
  (`RETRAIN_EVERY`, padrão 200), sem derrubar as requisições: o modelo novo
  só substitui o antigo depois de pronto.

## 7. Avaliação offline

`python -m scripts.avaliar` (não precisa do banco nem da API).

**Protocolo:** para cada usuário, 20% das avaliações (sorteadas) vão para o
teste e o modelo treina com os outros 80%. Para cada usuário, gera-se um top-10
com os filmes que ele ainda não viu no treino. São *relevantes* os filmes do
teste com nota ≥ 4. No cenário *cold start*, o modelo só enxerga 3 notas de
cada usuário.

**MovieLens latest-small** (610 usuários, 9.742 filmes, 100.836 avaliações),
todos os usuários, divisão com semente 42:

| Cenário | Estratégia | Precision@10 | Recall@10 | nDCG@10 | Cobertura |
|:--|:--|--:|--:|--:|--:|
| Normal | popularidade | 0,1158 | 0,0922 | 0,1509 | 0,0052 |
| Normal | conteúdo | 0,0047 | 0,0051 | 0,0053 | 0,2213 |
| Normal | colaborativa | **0,1733** | **0,1475** | **0,2283** | 0,0594 |
| Normal | **híbrido** | 0,1726 | 0,1472 | 0,2274 | 0,0510 |
| Cold start (3 notas) | popularidade | **0,1158** | **0,0922** | **0,1509** | 0,0052 |
| Cold start (3 notas) | conteúdo | 0,0055 | 0,0037 | 0,0060 | 0,3394 |
| Cold start (3 notas) | colaborativa | 0,0548 | 0,0501 | 0,0746 | 0,1683 |
| Cold start (3 notas) | **híbrido** | 0,1015 | 0,0823 | 0,1314 | 0,0473 |

Com outra semente (7), os números ficam praticamente iguais (híbrido:
0,170 no cenário normal e 0,104 no cold start), então o ajuste não ficou
preso a uma divisão específica.

**Leitura dos resultados:**

- **Com histórico**, o híbrido empata com a colaborativa (≈ 0,173) e fica
  **50% acima** da popularidade (0,116).
- **Com só 3 notas**, a colaborativa pura despenca para 0,055. O híbrido
  segura 0,10, quase o dobro, porque o α baixo passa o peso para
  conteúdo + popularidade.
- **O híbrido é a única estratégia que vai bem nos dois cenários.** É por
  isso que ele é o padrão.
- **No cold start extremo, a popularidade ainda ganha** (0,116 contra 0,102).
  Esse é um resultado conhecido na literatura: no MovieLens, os filmes
  populares concentram muitas avaliações altas, e com 3 notas há pouca
  informação para personalizar. A diferença está na cobertura: a
  popularidade recomenda a mesma meia dúzia de filmes para todo mundo
  (0,5% do catálogo), enquanto o híbrido já personaliza (4,7%).
- **Conteúdo sozinho é fraco aqui**, porque os metadados do MovieLens são
  pobres (19 gêneros, e tags em só ~16% dos filmes). Ele serve como
  complemento, para itens novos e para usuários que só declararam gêneros.

### Ajuste de hiperparâmetros

Os valores padrão saíram de uma busca em grade numa amostra de 250 usuários:

| Parâmetro | Testado | Escolhido |
|---|---|---|
| Score da colaborativa | média ponderada, soma centralizada, soma + bônus, implícito | soma + bônus 0,5 |
| K vizinhos | 50, 200 | 50 |
| K do α (`HYBRID_K`) | 3, 10 | 10 |
| Peso da popularidade | 0,1, 0,3 | 0,1 |
| Mistura conteúdo/popularidade | 0,5, 0,8 | 0,8 |

Todos podem ser trocados por variável de ambiente (ver README).

## 8. Decisões de arquitetura

- **O núcleo do modelo não depende do banco nem do FastAPI**
  (`app/recommender/`): ele recebe DataFrames e devolve recomendações. Isso
  permite testá-lo isoladamente e avaliá-lo offline com o mesmo código que
  roda na API.
- **Uma instância do modelo em memória** (`app/services/recommender_service.py`).
  O treino monta um modelo novo e só então troca a referência, com um lock;
  as requisições em andamento continuam usando o modelo antigo.
- **Notas lidas do banco a cada recomendação**, o que garante que o resultado
  reflete o estado atual sem precisar retreinar.
- **PostgreSQL no Docker e SQLite local/testes**, com o mesmo código
  (SQLAlchemy 2). A carga corrige as *sequences* do Postgres depois de
  inserir os IDs do MovieLens; sem isso, o primeiro `POST /usuarios` daria
  conflito de chave.
- **Carga idempotente:** o container pode reiniciar à vontade, porque o seed
  só carrega se o banco estiver vazio.
- **API em português** (rotas, campos e mensagens), porque é o que aparece no
  Swagger para quem vai usar.

## 9. Limitações e próximos passos

- A similaridade item-item é recalculada inteira a cada treino. Para o
  MovieLens 25M, seria preciso usar fatoração de matriz (ALS) ou um índice de
  vizinhos aproximados (FAISS).
- Uma divisão temporal (treinar no passado e testar no futuro) seria mais
  realista que a aleatória.
- O conteúdo poderia usar sinopses e embeddings de texto (por exemplo, via
  TMDb, usando o `links.csv` do MovieLens), o que deve melhorar bastante o
  cold start.
- Não há autenticação: qualquer cliente pode avaliar por qualquer usuário.
