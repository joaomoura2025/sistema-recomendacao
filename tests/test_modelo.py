"""Testes unitários do núcleo de recomendação (sem banco, sem API)."""

import numpy as np
import pandas as pd
import pytest

from app.recommender import ContentModel, HybridRecommender, ItemKNN, PopularityModel, Strategy
from app.recommender.content import split_genres
from tests.conftest import COMEDY, ROMANCE, SCIFI


# --------------------------------------------------------------- conteúdo
class TestConteudo:
    def test_split_genres(self):
        assert split_genres("Action|Sci-Fi") == ["Action", "Sci-Fi"]
        assert split_genres("(no genres listed)") == []
        assert split_genres(None) == []
        assert split_genres(["Drama", ""]) == ["Drama"]

    def test_vetores_normalizados(self, items_df):
        m = ContentModel().fit(items_df)
        norms = np.sqrt(np.asarray(m.matrix.multiply(m.matrix).sum(axis=1)).ravel())
        assert np.allclose(norms, 1.0)

    def test_similares_sao_do_mesmo_genero(self, items_df):
        m = ContentModel().fit(items_df)
        similares = [i for i, _ in m.similar(1, n=3)]
        assert set(similares) <= set(SCIFI)
        assert 1 not in similares

    def test_genero_favorito_resolve_cold_start(self, items_df):
        m = ContentModel().fit(items_df)
        perfil = m.user_profile({}, user_mean=3.5, favorite_genres=["Romance"])
        top4 = m.item_ids[np.argsort(-m.score(perfil))[:4]]
        assert set(top4) == set(ROMANCE)

    def test_perfil_vazio(self, items_df):
        m = ContentModel().fit(items_df)
        assert m.user_profile({}, 3.5) is None
        assert not m.score(None).any()


# ----------------------------------------------------------- colaborativa
class TestColaborativa:
    def test_vizinhos_corretos(self, train_df):
        knn = ItemKNN(k=5, shrinkage=0, min_item_ratings=1).fit(train_df)
        vizinhos = [i for i, _ in knn.similar(1, n=3)]
        assert set(vizinhos) <= set(SCIFI)

    def test_score_prioriza_genero_que_o_usuario_curtiu(self, train_df):
        knn = ItemKNN(k=5, min_item_ratings=1).fit(train_df)
        scores = knn.score({1: 5.0, 2: 5.0, 5: 1.0}, user_mean=3.67)
        s = dict(zip(knn.item_ids, scores))
        assert min(s[3], s[4]) > max(s[i] for i in ROMANCE)

    def test_shrinkage_reduz_similaridade(self, train_df):
        sem = ItemKNN(shrinkage=0, min_item_ratings=1).fit(train_df)
        com = ItemKNN(shrinkage=50, min_item_ratings=1).fit(train_df)
        assert com.sim.sum() < sem.sim.sum()

    def test_min_item_ratings_filtra_itens_raros(self, train_df):
        raro = pd.DataFrame([(1, 99, 5.0)], columns=["user_id", "item_id", "rating"])
        knn = ItemKNN(min_item_ratings=2).fit(pd.concat([train_df, raro]))
        assert 99 not in knn.index

    def test_sem_avaliacoes(self, train_df):
        knn = ItemKNN().fit(train_df)
        assert not knn.score({}, 3.5).any()
        vazio = ItemKNN().fit(pd.DataFrame(columns=["user_id", "item_id", "rating"]))
        assert vazio.n_items == 0

    def test_explicacao_aponta_filme_avaliado(self, train_df):
        knn = ItemKNN(min_item_ratings=1).fit(train_df)
        assert knn.top_contributor(3, {1: 5.0, 5: 1.0}, user_mean=3.0) == 1


# ------------------------------------------------------------ popularidade
def test_media_bayesiana_desconfia_de_poucos_votos():
    df = pd.DataFrame(
        # filme 1: 50 notas 4.5 | filme 2: uma única nota 5 | filme 3: 50 notas 2
        [(u, 1, 4.5) for u in range(50)] + [(0, 2, 5.0)] + [(u, 3, 2.0) for u in range(50)],
        columns=["user_id", "item_id", "rating"],
    )
    pop = PopularityModel(c=20).fit(df)
    assert pop.scores[1] > pop.scores[2]


# ----------------------------------------------------------------- híbrido
@pytest.fixture
def modelo(items_df, train_df):
    return HybridRecommender(min_item_ratings=1).fit(items_df, train_df)


class TestHibrido:
    def test_alpha_cresce_com_historico(self, modelo):
        assert modelo.alpha({}) == 0
        assert modelo.alpha({1: 5, 2: 5}) == pytest.approx(2 / 12)
        assert modelo.alpha({i: 4 for i in range(1, 11)}) == pytest.approx(0.5)

    @pytest.mark.parametrize("estrategia", list(Strategy))
    def test_nao_recomenda_o_que_ja_foi_avaliado(self, modelo, estrategia):
        avaliados = {1: 5.0, 2: 4.5, 5: 1.0}
        recs = modelo.recommend(avaliados, n=5, strategy=estrategia)
        assert recs
        assert not {r.item_id for r in recs} & set(avaliados)

    def test_fa_de_scifi_recebe_scifi(self, modelo):
        recs = modelo.recommend({1: 5.0, 2: 5.0, 5: 1.0, 6: 1.0}, n=2)
        assert {r.item_id for r in recs} == {3, 4}

    def test_usuario_novo_sem_nada_recebe_populares(self, modelo):
        recs = modelo.recommend({}, n=3)
        assert len(recs) == 3
        assert all(r.reason.startswith("Popular") for r in recs)

    def test_usuario_novo_com_genero_favorito(self, modelo):
        recs = modelo.recommend({}, favorite_genres=["Comedy"], n=4)
        assert {r.item_id for r in recs} == set(COMEDY)
        assert "Comedy" in recs[0].reason

    def test_filtro_de_genero(self, modelo):
        recs = modelo.recommend({1: 5.0}, n=10, genre="Romance")
        assert {r.item_id for r in recs} <= set(ROMANCE)

    def test_diversidade_mistura_generos(self, modelo):
        sem = modelo.recommend({1: 5.0}, n=4, diversity=0.0)
        com = modelo.recommend({1: 5.0}, n=4, diversity=0.9)
        generos = lambda recs: {modelo.content.genres[modelo.content.index[r.item_id]][0] for r in recs}  # noqa: E731
        assert len(com) == 4 and len({r.item_id for r in com}) == 4
        assert len(generos(com)) >= len(generos(sem))
        assert len(generos(com)) >= 2

    def test_explicacao_colaborativa_cita_titulo(self, modelo):
        historico = {i: 5.0 for i in (1, 2)} | {i: 1.0 for i in (5, 6, 7)} | {i: 3.0 for i in COMEDY}
        recs = modelo.recommend(historico, n=2)
        assert any("Space Movie" in r.reason for r in recs)

    def test_item_novo_entra_pelo_conteudo(self, modelo, items_df):
        novo = pd.DataFrame(
            [(100, "New Space Movie", "Sci-Fi|Action", 2024, "space aliens spaceship")],
            columns=items_df.columns,
        )
        modelo.fit_content(pd.concat([items_df, novo]))
        recs = modelo.recommend({1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0}, n=12, strategy=Strategy.CONTENT)
        assert 100 in [r.item_id for r in recs[:3]]
        assert modelo.similar_items(100, 3)[0][2] == "conteudo"

    def test_modelo_vazio(self):
        assert HybridRecommender().recommend({1: 5}) == []
