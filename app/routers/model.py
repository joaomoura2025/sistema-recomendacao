"""Endpoints de administração do modelo."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import schemas
from app.database import get_db
from app.services.recommender_service import service

router = APIRouter(prefix="/modelo", tags=["Modelo"])


@router.get("/status", response_model=schemas.StatusModelo)
def model_status():
    """Quando o modelo foi treinado, com quantos dados e quais parâmetros."""
    return service.status()


@router.post("/retreinar", response_model=schemas.StatusModelo)
def retrain(db: Session = Depends(get_db)):
    """Retreina o modelo inteiro com os dados atuais do banco (síncrono).
    No MovieLens small leva menos de 1 segundo."""
    service.train(db)
    return service.status()
