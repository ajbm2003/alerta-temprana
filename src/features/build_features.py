"""Feature engineering temporalmente consistente (anti-leakage).

Regla estricta: una feature del ano t solo usa informacion de anos <= t.
"""
import pandas as pd
import numpy as np

FEATURES_COLS = [
    "tasa_aprobacion_asignaturas",
    "tasa_desaprobacion_asignaturas",
    "tasa_ausencias_finales",
    "porcentaje_avance_carrera",
    "porcentaje_avance_cursado",
    "cantidad_pausas",
    "maxima_duracion_pausa",
    "antiguedad_carrera",
    "edad",
    "delta_tasa_aprobacion",
    "brecha_cursado_carrera",
    "ritmo_vs_plan",
    "anios_sin_finales",
]

# Los notebooks 01–08 conservan su selección original. La candidata final
# agrega estos tres recuentos, exactamente como en los notebooks 09–11.
RECUENTOS_COLS = ["asignaturas_cursadas", "inscripciones_finales", "finales_aprobados"]
FEATURES_MODELO_FINAL = FEATURES_COLS + RECUENTOS_COLS


def construir_target(df: pd.DataFrame, ventana: int = 2, censurar_hasta: int = 2025) -> pd.DataFrame:
    """abandono = sin actividad en los 'ventana' anos siguientes a t.

    Observaciones con ventana incompleta (ultimas cohortes) se excluyen.
    """
    df = df.sort_values(["id_alumno", "anio_academico"]).copy()
    df["prox_actividad"] = df.groupby("id_alumno")["anio_academico"].shift(-1)
    df["abandono"] = (
        (df["prox_actividad"].isna()) |
        (df["prox_actividad"] > df["anio_academico"] + ventana)
    ).astype(int)
    df.loc[df["anio_academico"] > censurar_hasta - ventana, "abandono"] = pd.NA
    return df.dropna(subset=["abandono"])


def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas; todas usan solo informacion <= t (ver configs/features.yaml)."""
    df = df.sort_values(["id_alumno", "anio_academico"]).copy()
    g = df.groupby("id_alumno")
    df["delta_tasa_aprobacion"] = g["tasa_aprobacion_asignaturas"].diff().fillna(0)
    df["brecha_cursado_carrera"] = df["porcentaje_avance_cursado"] - df["porcentaje_avance_carrera"]
    ritmo_teorico = (df["antiguedad_carrera"].astype(float) / 4).replace(0, np.nan)
    df["ritmo_vs_plan"] = (df["porcentaje_avance_carrera"] / ritmo_teorico).fillna(0)
    aprobo_final = (df["finales_aprobados"] > 0).astype(int)
    sin_finales = aprobo_final.groupby(df["id_alumno"]).apply(
        lambda s: s.eq(0).groupby(s.cumsum()).cumsum()
    )
    df["anios_sin_finales"] = sin_finales.reset_index(level=0, drop=True)
    df[FEATURES_COLS] = df[FEATURES_COLS].fillna(0)
    return df


def matriz_modelado(df: pd.DataFrame):
    X = df[FEATURES_COLS]
    y = df["abandono"].astype(int)
    groups = df["id_alumno"]
    return X, y, groups
