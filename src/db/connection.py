"""Conexión PostgreSQL y soporte SQLite para una demo local aislada."""
import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from src.data.load_data import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env", override=False)


def crear_engine(url=None):
    """Construye un engine sin abrir todavía una conexión.

    DATABASE_URL se respeta, incluidas sus opciones de SSL. No se imprime
    la URL para evitar exponer contraseñas en logs.
    """
    if url is None:
        url = os.environ.get("DATABASE_URL")
    if url is None:
        url = URL.create(
            "postgresql+psycopg2",
            username=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres"),
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", "5432")),
            database=os.environ.get("DB_NAME", "alerta"),
        )
    if isinstance(url, str) and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    parsed = make_url(url)
    # SQLAlchemy 2.1 elige psycopg 3 por defecto; el proyecto instala psycopg2.
    if parsed.drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+psycopg2")
    opciones = {"pool_pre_ping": True}
    if parsed.get_backend_name() == "sqlite":
        opciones["connect_args"] = {"check_same_thread": False}
        if parsed.database in (None, "", ":memory:"):
            opciones["poolclass"] = StaticPool

    resultado = create_engine(parsed, **opciones)
    if parsed.get_backend_name() == "sqlite":
        # La demo también comprueba las claves foráneas, no solo las columnas.
        @event.listens_for(resultado, "connect")
        def activar_claves_foraneas(dbapi_connection, _):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return resultado


# Crear el engine no consulta ni modifica la base.
engine = crear_engine()
