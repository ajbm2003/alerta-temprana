import os
# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from sqlalchemy import text
from src.db.connection import engine

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_DB_TESTS", "").lower() in {"1", "true", "yes"},
    reason="sin base de datos disponible",
)


def test_conexion_y_tablas():
    esperadas = {"estudiantes", "historico_anual", "predicciones",
                 "metricas_modelo", "drift"}
    with engine.connect() as conn:
        res = conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        tablas = {r[0] for r in res}
    assert esperadas.issubset(tablas)
