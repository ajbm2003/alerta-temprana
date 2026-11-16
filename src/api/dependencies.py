"""Dependencias pequeñas para cargar una vez el modelo y probar sin servicios."""
import hmac
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.models.predict import ModeloNoDisponible, banda, cargar_paquete
from src.data.load_data import PROJECT_ROOT

# También permite que el primer request sea /score-batch con token de .env.
load_dotenv(PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_paquete():
    """Mantiene el paquete cargado; no entrena ni descarga en cada predicción."""
    try:
        return cargar_paquete()
    except ModeloNoDisponible as exc:
        logger.warning("El paquete de modelo no está disponible: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Revisá la ruta local o el repositorio configurado.",
        ) from exc


def get_modelo():
    """Conserva el acceso al estimador y a las columnas para otros clientes."""
    paquete = get_paquete()
    return paquete["modelo"], paquete["features"]


def get_db_engine():
    """La API individual no necesita importar ni conectar PostgreSQL."""
    from src.db.connection import engine
    return engine


def verificar_token(
    credenciales: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    """Solo el endpoint que escribe el batch exige el token configurado."""
    esperado = os.environ.get("API_TOKEN")
    if not esperado:
        raise HTTPException(status_code=503, detail="API_TOKEN no está configurado para el batch.")
    if (
        credenciales is None
        or credenciales.scheme.lower() != "bearer"
        or not hmac.compare_digest(credenciales.credentials, esperado)
    ):
        raise HTTPException(
            status_code=401,
            detail="Token de batch inválido.",
            headers={"WWW-Authenticate": "Bearer"},
        )
