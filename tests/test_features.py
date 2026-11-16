"""Test CRITICO anti-leakage: las features del ano t no pueden conocer t+1."""
import pandas as pd
import numpy as np
from src.features.build_features import agregar_features, construir_target


def _df_sintetico():
    rng = np.random.default_rng(0)
    filas = []
    for alumno in range(1, 6):
        for anio in range(2018, 2023):
            filas.append({
                "id_alumno": alumno, "anio_academico": anio, "edad": 22 + anio - 2018,
                "asignaturas_cursadas": 5, "asignaturas_aprobadas": 4,
                "finales_aprobados": 3, "tasa_aprobacion_asignaturas": 0.8,
                "tasa_ausencias_finales": 0.1, "porcentaje_avance_carrera": 0.3,
                "porcentaje_avance_cursado": 0.4, "cantidad_pausas": 0,
                "maxima_duracion_pausa": 0, "antiguedad_carrera": anio - 2018,
                "cambio_plan": 0, "tasa_desaprobacion_asignaturas": 0.1,
                "finales_ausentes": 0, "equivalencias_asignaturas": 0,
            })
    return pd.DataFrame(filas)


def test_feature_no_depende_de_anios_posteriores():
    """Al duplicar los datos de 2022 con valores extremos en 2018,
    las features de 2018 no deben cambiar (no miran al futuro)."""
    df = _df_sintetico()
    f1 = agregar_features(df)
    df_mod = df.copy()
    df_mod.loc[df_mod["anio_academico"] == 2022, "tasa_aprobacion_asignaturas"] = 0.0
    f2 = agregar_features(df_mod)
    cols = ["delta_tasa_aprobacion", "brecha_cursado_carrera",
            "ritmo_vs_plan", "anios_sin_finales"]
    assert f1.loc[f1["anio_academico"] == 2018, cols].reset_index(drop=True).equals(
        f2.loc[f2["anio_academico"] == 2018, cols].reset_index(drop=True))


def test_censurado_excluido():
    df = _df_sintetico()
    df.loc[df["anio_academico"] == 2022, "anio_academico"] = 2025
    out = construir_target(df, ventana=2, censurar_hasta=2025)
    assert (out["anio_academico"] <= 2023).all()
