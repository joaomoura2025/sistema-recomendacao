"""Tabelas do banco: usuários, itens (filmes) e avaliações."""

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120))
    # Gêneros favoritos separados por "|" (mesmo formato do MovieLens).
    # Usados no "cold start", quando o usuário ainda não avaliou nada.
    favorite_genres: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def favorite_genres_list(self) -> list[str]:
        return [g for g in (self.favorite_genres or "").split("|") if g]


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    genres: Mapped[str] = mapped_column(String(500), default="")  # "Action|Comedy"
    year: Mapped[int | None] = mapped_column(Integer)
    # Texto livre usado pela filtragem baseada em conteúdo: tags do MovieLens
    # ou a descrição informada ao cadastrar um item novo.
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )

    @property
    def genres_list(self) -> list[str]:
        return [g for g in (self.genres or "").split("|") if g]


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "item_id", name="uq_rating_user_item"),
        CheckConstraint("rating >= 0.5 AND rating <= 5", name="ck_rating_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), index=True)
    rating: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped[User] = relationship(back_populates="ratings")
    item: Mapped[Item] = relationship(back_populates="ratings")
