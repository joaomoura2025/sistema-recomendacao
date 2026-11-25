"""Testes de integração: API completa (FastAPI + SQLAlchemy + modelo)."""

import dataclasses

import pytest

from app.services import recommender_service
from tests.conftest import COMEDY, ROMANCE, SCIFI

pytestmark = pytest.mark.integracao

JSON = {"content-type": "application/json"}


def ids(resp_or_body) -> list[int]:
    """IDs recomendados, a partir da resposta HTTP ou do JSON já lido."""
    body = resp_or_body if isinstance(resp_or_body, dict) else resp_or_body.json()
    return [r["item"]["id"] for r in body["recomendacoes"]]


# ------------------------------------------------------------------ sistema
def test_saude_e_docs(client):
    assert client.get("/saude").json() == {"status": "ok", "modelo_treinado": True}
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/usuarios/{user_id}/recomendacoes" in paths


def test_generos(client):
    assert set(client.get("/generos").json()) == {"Action", "Comedy", "Drama", "Romance", "Sci-Fi"}


# ----------------------------------------------------------------- usuários
def test_criar_usuario_normaliza_generos(client):
    r = client.post("/usuarios", json={"nome": "Ana", "generos_favoritos": ["sci-fi", "SCI-FI"]})
    assert r.status_code == 201
    body = r.json()
    assert body["generos_favoritos"] == ["Sci-Fi"]
    assert body["total_avaliacoes"] == 0
    assert client.get(f"/usuarios/{body['id']}").json()["nome"] == "Ana"


def test_genero_invalido(client):
    r = client.post("/usuarios", json={"generos_favoritos": ["Faroeste Espacial"]})
    assert r.status_code == 422
    assert "desconhecido" in r.json()["detail"]


def test_usuario_inexistente(client):
    assert client.get("/usuarios/9999").status_code == 404
    assert client.get("/usuarios/9999/recomendacoes").status_code == 404


def test_atualizar_preferencias_muda_cold_start(client):
    uid = client.post("/usuarios", json={"generos_favoritos": ["Romance"]}).json()["id"]
    assert set(ids(client.get(f"/usuarios/{uid}/recomendacoes?n=4"))) == set(ROMANCE)

    r = client.patch(f"/usuarios/{uid}/preferencias", json={"generos_favoritos": ["Comedy"]})
    assert r.status_code == 200 and r.json()["generos_favoritos"] == ["Comedy"]
    assert set(ids(client.get(f"/usuarios/{uid}/recomendacoes?n=4"))) == set(COMEDY)


# --------------------------------------------------------------- avaliações
def test_avaliar_cria_e_depois_atualiza(client):
    uid = client.post("/usuarios", json={}).json()["id"]
    r1 = client.put(f"/usuarios/{uid}/avaliacoes/1", json={"nota": 4})
    r2 = client.put(f"/usuarios/{uid}/avaliacoes/1", json={"nota": 2.5})
    assert (r1.status_code, r2.status_code) == (201, 200)
    lista = client.get(f"/usuarios/{uid}/avaliacoes").json()
    assert len(lista) == 1 and lista[0]["nota"] == 2.5 and lista[0]["titulo"] == "Space Movie 1"


@pytest.mark.parametrize("nota", [0, 5.5, 3.3, "abc"])
def test_nota_invalida(client, nota):
    assert client.put("/usuarios/1/avaliacoes/1", json={"nota": nota}).status_code == 422


def test_avaliar_item_inexistente(client):
    assert client.put("/usuarios/1/avaliacoes/999", json={"nota": 4}).status_code == 404


def test_avaliacoes_em_lote_e_remocao(client):
    uid = client.post("/usuarios", json={}).json()["id"]
    lote = {"avaliacoes": [{"item_id": 1, "nota": 5}, {"item_id": 2, "nota": 4.5}]}
    assert client.post(f"/usuarios/{uid}/avaliacoes", json=lote).json() == {"criadas": 2, "atualizadas": 0}
    assert client.post(f"/usuarios/{uid}/avaliacoes", json=lote).json() == {"criadas": 0, "atualizadas": 2}

    assert client.delete(f"/usuarios/{uid}/avaliacoes/1").status_code == 204
    assert client.delete(f"/usuarios/{uid}/avaliacoes/1").status_code == 404
    assert client.get(f"/usuarios/{uid}").json()["total_avaliacoes"] == 1


# ------------------------------------------------------------ recomendações
def test_recomendacoes_reagem_na_hora_as_avaliacoes(client):
    """Sem retreinar: as notas novas já mudam o resultado."""
    uid = client.post("/usuarios", json={}).json()["id"]
    antes = client.get(f"/usuarios/{uid}/recomendacoes?n=3").json()
    assert antes["peso_colaborativa"] == 0
    assert all(r["motivo"].startswith("Popular") for r in antes["recomendacoes"])

    for item, nota in [(1, 5), (2, 5), (5, 1), (6, 1)]:
        client.put(f"/usuarios/{uid}/avaliacoes/{item}", json={"nota": nota})

    depois = client.get(f"/usuarios/{uid}/recomendacoes?n=2").json()
    assert set(ids(depois)) == {3, 4}
    assert depois["total_avaliacoes"] == 4
    assert 0 < depois["peso_colaborativa"] < 1
    rec = depois["recomendacoes"][0]
    assert set(rec["componentes"]) == {"colaborativa", "conteudo", "popularidade"}
    assert rec["motivo"]


def test_recomendacao_nunca_repete_item_avaliado(client):
    body = client.get("/usuarios/1/recomendacoes?n=20").json()
    avaliados = {a["item_id"] for a in client.get("/usuarios/1/avaliacoes").json()}
    assert not set(ids(body)) & avaliados


@pytest.mark.parametrize("estrategia", ["hibrido", "colaborativa", "conteudo", "popularidade"])
def test_estrategias(client, estrategia):
    r = client.get(f"/usuarios/1/recomendacoes?estrategia={estrategia}&n=3")
    assert r.status_code == 200
    assert r.json()["estrategia"] == estrategia


def test_parametros_invalidos(client):
    assert client.get("/usuarios/1/recomendacoes?estrategia=magica").status_code == 422
    assert client.get("/usuarios/1/recomendacoes?n=0").status_code == 422
    assert client.get("/usuarios/1/recomendacoes?diversidade=2").status_code == 422
    assert client.get("/usuarios/1/recomendacoes?genero=Faroeste").status_code == 422


def test_filtro_genero_e_diversidade(client):
    uid = client.post("/usuarios", json={}).json()["id"]
    client.put(f"/usuarios/{uid}/avaliacoes/1", json={"nota": 5})
    so_comedia = ids(client.get(f"/usuarios/{uid}/recomendacoes?genero=comedy&n=10"))
    assert so_comedia and set(so_comedia) <= set(COMEDY)
    diverso = ids(client.get(f"/usuarios/{uid}/recomendacoes?diversidade=0.8&n=5"))
    assert len(set(diverso)) == 5


# ------------------------------------------------------------------- itens
def test_criar_item_ja_entra_nas_recomendacoes(client):
    r = client.post(
        "/itens",
        json={"titulo": "Brand New Space Movie", "generos": ["sci-fi", "Action"], "ano": 2024,
              "descricao": "Space aliens spaceship"},
    )
    assert r.status_code == 201
    novo = r.json()
    assert novo["generos"] == ["Sci-Fi", "Action"] and novo["total_avaliacoes"] == 0

    uid = client.post("/usuarios", json={}).json()["id"]
    lote = {"avaliacoes": [{"item_id": i, "nota": 5} for i in SCIFI]}
    client.post(f"/usuarios/{uid}/avaliacoes", json=lote)
    recs = ids(client.get(f"/usuarios/{uid}/recomendacoes?estrategia=conteudo&n=1"))
    assert recs == [novo["id"]]

    sim = client.get(f"/itens/{novo['id']}/similares?n=3").json()
    assert {s["item"]["id"] for s in sim} <= set(SCIFI)


def test_item_com_genero_novo_e_aceito(client):
    r = client.post("/itens", json={"titulo": "Doc", "generos": ["Documentary"]})
    assert r.status_code == 201
    assert "Documentary" in client.get("/generos").json()


def test_listar_e_buscar_itens(client):
    assert client.get("/itens?limite=5").json()["total"] == 12
    busca = client.get("/itens?busca=love").json()
    assert busca["total"] == 4
    assert client.get("/itens?genero=Comedy").json()["total"] == 4
    item = client.get("/itens/1").json()
    assert item["titulo"] == "Space Movie 1" and item["total_avaliacoes"] > 0
    assert client.get("/itens/999").status_code == 404


def test_similares_colaborativos(client):
    sim = client.get("/itens/5/similares?n=3").json()
    assert len(sim) == 3
    assert {s["item"]["id"] for s in sim} <= set(ROMANCE)
    assert sim[0]["fonte"] == "colaborativa"


# ------------------------------------------------------------------ modelo
def test_status_e_retreino(client):
    status = client.get("/modelo/status").json()
    assert status["itens"] == 12 and status["avaliacoes_no_treino"] > 0

    uid = client.post("/usuarios", json={}).json()["id"]
    client.put(f"/usuarios/{uid}/avaliacoes/1", json={"nota": 5})
    assert client.get("/modelo/status").json()["alteracoes_pendentes"] == 1

    novo = client.post("/modelo/retreinar").json()
    assert novo["alteracoes_pendentes"] == 0
    assert novo["avaliacoes_no_treino"] == status["avaliacoes_no_treino"] + 1


def test_retreino_automatico(client, monkeypatch):
    monkeypatch.setattr(
        recommender_service, "settings",
        dataclasses.replace(recommender_service.settings, retrain_every=3),
    )
    antes = client.get("/modelo/status").json()["treinado_em"]
    uid = client.post("/usuarios", json={}).json()["id"]
    for item in (1, 2, 3):
        client.put(f"/usuarios/{uid}/avaliacoes/{item}", json={"nota": 4})
    status = client.get("/modelo/status").json()
    assert status["treinado_em"] != antes
    assert status["alteracoes_pendentes"] == 0
