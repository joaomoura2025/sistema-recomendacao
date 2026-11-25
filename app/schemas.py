"""Schemas (Pydantic) de entrada e saída da API.

Os campos estão em português porque são o que aparece no Swagger.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.recommender import Strategy


class Mensagem(BaseModel):
    detail: str


# ---------------------------------------------------------------- usuários
class UsuarioCriar(BaseModel):
    nome: str | None = Field(None, max_length=120, examples=["João"])
    generos_favoritos: list[str] = Field(
        default_factory=list,
        description="Gêneros do MovieLens (ver GET /generos). "
        "Usados enquanto o usuário tem poucas avaliações.",
        examples=[["Sci-Fi", "Thriller"]],
    )


class PreferenciasAtualizar(BaseModel):
    nome: str | None = Field(None, max_length=120)
    generos_favoritos: list[str] | None = Field(None, examples=[["Comedy", "Romance"]])


class Usuario(BaseModel):
    id: int
    nome: str | None
    generos_favoritos: list[str]
    total_avaliacoes: int
    criado_em: datetime | None


# ------------------------------------------------------------------- itens
class ItemCriar(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=300, examples=["Duna: Parte Dois"])
    generos: list[str] = Field(default_factory=list, examples=[["Sci-Fi", "Adventure"]])
    ano: int | None = Field(None, ge=1870, le=2100, examples=[2024])
    descricao: str = Field(
        "",
        max_length=5000,
        description="Texto livre (sinopse, palavras-chave). Entra na filtragem por conteúdo.",
        examples=["desert planet epic space opera sequel"],
    )


class ItemResumo(BaseModel):
    id: int
    titulo: str
    generos: list[str]
    ano: int | None


class Item(ItemResumo):
    descricao: str
    media_notas: float | None
    total_avaliacoes: int


class ListaItens(BaseModel):
    total: int
    itens: list[ItemResumo]


# -------------------------------------------------------------- avaliações
def _validar_nota(v: float) -> float:
    if v * 2 != int(v * 2):
        raise ValueError("a nota deve ser múltipla de 0.5 (ex.: 3.5)")
    return v


class NotaEntrada(BaseModel):
    nota: float = Field(..., ge=0.5, le=5.0, examples=[4.5])

    _v = field_validator("nota")(_validar_nota)


class AvaliacaoEntrada(NotaEntrada):
    item_id: int = Field(..., examples=[1])


class AvaliacoesLote(BaseModel):
    avaliacoes: list[AvaliacaoEntrada] = Field(..., min_length=1, max_length=500)


class Avaliacao(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    item_id: int
    titulo: str
    nota: float
    data: datetime


class ResultadoLote(BaseModel):
    criadas: int
    atualizadas: int


# ---------------------------------------------------------- recomendações
class Componentes(BaseModel):
    colaborativa: float
    conteudo: float
    popularidade: float


class Recomendacao(BaseModel):
    item: ItemResumo
    score: float
    componentes: Componentes
    motivo: str


class RespostaRecomendacoes(BaseModel):
    usuario_id: int
    estrategia: Strategy
    total_avaliacoes: int
    peso_colaborativa: float = Field(
        ..., description="α = n/(n+K): quanto a filtragem colaborativa pesa para este usuário"
    )
    recomendacoes: list[Recomendacao]


class ItemSimilar(BaseModel):
    item: ItemResumo
    similaridade: float
    fonte: str


# -------------------------------------------------------------------- modelo
class StatusModelo(BaseModel):
    treinado_em: datetime | None
    duracao_treino_s: float | None
    itens: int
    itens_colaborativa: int
    avaliacoes_no_treino: int
    alteracoes_pendentes: int
    retreino_automatico_a_cada: int
    parametros: dict
