"""Pruebas de recorrido: historia → features → modelo → API/base de datos."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect, text

from scripts.cargar_datos import main as cargar_datos
from scripts.predict_batch import main as predict_batch
from src.data.load_data import PROJECT_ROOT, load_raw
from src.db.queries import leer, ultimas_predicciones
from src.db.schema import inicializar_db
from src.features.build_features import agregar_features
from src.models.predict import cargar_paquete, predecir_historial, preparar_historial
from src.models.train import entrenar


def test_features_coinciden_con_calculo_directo(historial_sintetico):
    preparado = preparar_historial(historial_sintetico.iloc[::-1], 2021)
    esperado = agregar_features(
        historial_sintetico.sort_values(["id_alumno","anio_academico"]).reset_index(drop=True)
    )
    columnas = [
        "id_alumno", "anio_academico", "delta_tasa_aprobacion",
        "brecha_cursado_carrera", "ritmo_vs_plan", "anios_sin_finales",
    ]
    pd.testing.assert_frame_equal(preparado[columnas], esperado[columnas])


def test_base_batch_y_api_coinciden_y_reintentar_no_duplica(
    db_engine, historial_sintetico, paquete_sintetico, api_client, payload,
):
    for _ in range(2):
        cargar_datos(
            engine=db_engine, datos=historial_sintetico,
            anio_corte=2021, incluir_metricas=False,
        )
    assert len(leer("historico_anual", engine=db_engine)) == len(historial_sintetico)

    respuesta = api_client.post(
        "/score-batch", headers={"Authorization": "Bearer token-de-prueba"}
    )
    assert respuesta.status_code == 200, respuesta.text
    assert respuesta.json()["registros_scorados"] == 6
    predict_batch(engine=db_engine, paquete=paquete_sintetico, anio_corte=2021)

    guardadas = leer("predicciones", engine=db_engine)
    assert len(guardadas) == 6
    assert guardadas.anio_base.eq(2021).all()
    assert guardadas.factor_1.notna().all()
    assert len(ultimas_predicciones(engine=db_engine)) == 6

    api = api_client.post("/predict", json=payload).json()
    fila = guardadas.loc[guardadas.id_alumno.eq(1)].iloc[0]
    assert abs(api["probabilidad_abandono"] - fila.probabilidad_abandono) <= 5.1e-7
    assert api["banda_riesgo"] == fila.banda_riesgo
    assert api["factores_principales"][0] == fila.factor_1


def test_migracion_no_borra_historia_anterior():
    from src.db.connection import crear_engine

    db = crear_engine("sqlite:///:memory:")
    try:
        with db.begin() as conexion:
            conexion.execute(text(
                "CREATE TABLE historico_anual (id_alumno INTEGER, "
                "anio_academico INTEGER, edad INTEGER, "
                "PRIMARY KEY (id_alumno, anio_academico))"
            ))
            conexion.execute(text(
                "INSERT INTO historico_anual VALUES (1, 2020, 25)"
            ))
        inicializar_db(db)
        columnas = {c["name"] for c in inspect(db).get_columns("historico_anual")}
        assert "inscripciones_finales" in columnas
        with db.connect() as conexion:
            fila = conexion.execute(text(
                "SELECT edad, inscripciones_finales FROM historico_anual"
            )).one()
        assert fila == (25, None)
    finally:
        db.dispose()


ARCHIVOS_REALES = [
    PROJECT_ROOT / "models/modelo_final.joblib",
    PROJECT_ROOT / "reports/scoring_demostracion.csv",
    PROJECT_ROOT / "data/raw/Psicologia_2023_Anonimizado.xlsx",
    PROJECT_ROOT / "data/processed/dataset_modelado.parquet",
]


@pytest.mark.skipif(
    not all(p.exists() for p in ARCHIVOS_REALES),
    reason="Artefactos locales excluidos del repositorio.",
)
def test_modelo_y_batch_reproducen_los_663_casos_del_notebook(db_engine):
    paquete = cargar_paquete(ARCHIVOS_REALES[0])
    corte = paquete["metadata"]["target"]["cierre_seguimiento"]
    cargar_datos(engine=db_engine, anio_corte=corte)
    n = predict_batch(engine=db_engine, paquete=paquete, anio_corte=corte)

    guardadas = leer("predicciones", engine=db_engine)
    referencia = pd.read_csv(ARCHIVOS_REALES[1])
    comparacion = guardadas.merge(
        referencia,
        left_on=["id_alumno","anio_base"],
        right_on=["id_alumno","anio_academico"],
        suffixes=("_batch","_notebook"),
        validate="one_to_one",
    )
    assert n == len(comparacion) == len(referencia)
    np.testing.assert_allclose(
        comparacion.probabilidad_abandono_batch,
        comparacion.probabilidad_abandono_notebook,
        atol=1e-12, rtol=1e-12,
    )


@pytest.mark.skipif(
    not all(p.exists() for p in ARCHIVOS_REALES),
    reason="Artefactos locales excluidos del repositorio.",
)
def test_entrenamiento_reproduce_pipeline_final_sin_reconstruir_target(tmp_path):
    ruta = tmp_path / "reentrenado.joblib"
    resumen = entrenar(ruta_modelo=ruta)
    nuevo = cargar_paquete(ruta)
    previo = cargar_paquete(ARCHIVOS_REALES[0])
    historia = load_raw()

    esperado = predecir_historial(historia, previo, 2025)
    actual = predecir_historial(historia, nuevo, 2025)
    assert resumen["n_features"] == 16
    assert resumen["registros"] == 1979
    np.testing.assert_allclose(
        actual.probabilidad_abandono,
        esperado.probabilidad_abandono,
        atol=1e-12, rtol=1e-12,
    )
