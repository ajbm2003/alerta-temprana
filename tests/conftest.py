"""Fixtures pequeñas: el CI no necesita el Excel, el modelo ni Railway."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_db_engine, get_paquete
from src.api.main import app
from src.api.schemas import HistoriaAnual
from src.db.connection import crear_engine
from src.db.schema import inicializar_db
from src.features.build_features import FEATURES_MODELO_FINAL, agregar_features
from src.models.train import crear_pipeline


@pytest.fixture
def historial_sintetico():
    filas = []
    for alumno in range(1, 7):
        for anio in [2019, 2020, 2021]:
            aprobadas = 2 if alumno % 2 else 5
            filas.append({
                "id_alumno": alumno,
                "anio_academico": anio,
                "cohorte_carrera": 2019,
                "cohorte_universidad": 2019,
                "tuvo_actividad_previa": 0,
                "anios_actividad_previa": 0,
                "edad": 21 + alumno + anio - 2019,
                "antiguedad_carrera": anio - 2019,
                "asignaturas_cursadas": 6,
                "asignaturas_aprobadas": aprobadas,
                "inscripciones_finales": 3,
                "finales_aprobados": 1 if alumno % 2 else 3,
                "tasa_aprobacion_asignaturas": aprobadas / 6,
                "tasa_desaprobacion_asignaturas": (6 - aprobadas) / 6,
                "tasa_ausencias_finales": 0.3 if alumno % 2 else 0.0,
                "porcentaje_avance_carrera": 0.1 * (anio - 2018),
                "porcentaje_avance_cursado": 0.12 * (anio - 2018),
                "cantidad_pausas": 0,
                "maxima_duracion_pausa": 0,
                "finales_ausentes": 1 if alumno % 2 else 0,
                "equivalencias_asignaturas": 0,
                "cambio_plan": 1,
            })
    return pd.DataFrame(filas)


@pytest.fixture
def paquete_sintetico(historial_sintetico):
    features = agregar_features(historial_sintetico)
    X = features[FEATURES_MODELO_FINAL]
    y = features["id_alumno"].mod(2)
    modelo = crear_pipeline()
    modelo.fit(X, y)
    fondo = modelo[:-1].transform(X).mean(axis=0)
    return {
        "modelo": modelo,
        "features": list(FEATURES_MODELO_FINAL),
        "metadata": {
            "version": "fixture-logistica-v1",
            "features": list(FEATURES_MODELO_FINAL),
            "modelo": "logistica_recuentos",
            "target": {"cierre_seguimiento": 2021, "ventana_anios": 2},
            "calibracion": "sin_recalibracion_adicional",
            "bandas_riesgo_validadas": False,
            "umbral_validado": False,
            "umbral_referencia": 0.5,
            "explicabilidad": {"media_fondo_transformado": fondo.tolist()},
        },
    }


@pytest.fixture
def db_engine():
    engine = crear_engine("sqlite:///:memory:")
    inicializar_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def api_client(paquete_sintetico, db_engine, monkeypatch):
    anteriores = app.dependency_overrides.copy()
    monkeypatch.setenv("API_TOKEN", "token-de-prueba")
    monkeypatch.setenv("SCORING_YEAR", "2021")
    app.dependency_overrides[get_paquete] = lambda: paquete_sintetico
    app.dependency_overrides[get_db_engine] = lambda: db_engine
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    app.dependency_overrides.update(anteriores)


@pytest.fixture
def payload(historial_sintetico):
    historia = historial_sintetico.loc[
        historial_sintetico.id_alumno.eq(1), list(HistoriaAnual.model_fields)
    ].iloc[::-1].to_dict("records")
    return {"id_alumno": 1, "anio_corte": 2021, "historia": historia}
