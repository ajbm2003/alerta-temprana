"""Carga historial completo y métricas reales, sin usar el parquet censurado."""
import sys
from pathlib import Path

# Permite ejecutar "python scripts/cargar_datos.py" desde la raíz del proyecto.
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import numpy as np
import pandas as pd

from src.data.load_data import PROJECT_ROOT, load_history
from src.db.queries import cargar_historial, guardar_metricas
from src.models.predict import fecha_scoring, preparar_historial, resolver_corte
from src.models.train import VERSION_MODELO

EST_COLS = [
    "id_alumno", "cohorte_carrera", "cohorte_universidad",
    "tuvo_actividad_previa", "anios_actividad_previa",
]
HIST_COLS = [
    "id_alumno", "anio_academico", "edad",
    "asignaturas_cursadas", "asignaturas_aprobadas", "inscripciones_finales",
    "tasa_aprobacion_asignaturas", "tasa_desaprobacion_asignaturas",
    "finales_aprobados", "finales_ausentes", "tasa_ausencias_finales",
    "equivalencias_asignaturas", "porcentaje_avance_cursado",
    "porcentaje_avance_carrera", "cantidad_pausas",
    "maxima_duracion_pausa", "antiguedad_carrera", "cambio_plan",
]


def preparar_tablas(datos: pd.DataFrame, anio_corte: int):
    """Conserva todo el historial <= corte, sin construir el target."""
    df = datos.loc[datos["anio_academico"].le(anio_corte)].copy()
    faltantes = set(EST_COLS + HIST_COLS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas para la carga: {sorted(faltantes)}")
    if df.empty:
        raise ValueError("No hay datos para el año de corte.")

    # Usa la misma validación que la predicción, pero guarda columnas históricas.
    preparar_historial(df, anio_corte)
    columnas = list(dict.fromkeys(EST_COLS + HIST_COLS))
    df[columnas] = df[columnas].apply(pd.to_numeric, errors="raise")
    df = df.sort_values(["id_alumno", "anio_academico"])
    df["edad"] = df["edad"].astype(float).replace(0, np.nan)

    estudiantes = df[EST_COLS].drop_duplicates("id_alumno", keep="last")
    historico = df[HIST_COLS].copy()
    return estudiantes, historico


def preparar_metricas(ruta, modelo_version):
    """Pasa el CSV largo de bootstrap a una fila por año de evaluación.

    pr_auc conserva el nombre de la tabla original, pero almacena Average
    Precision, exactamente la métrica utilizada en los notebooks.
    """
    datos = pd.read_csv(ruta)
    equivalencias = {
        "roc_auc": "roc_auc", "average_precision": "pr_auc",
        "precision": "precision", "recall": "recall",
        "brier": "brier", "log_loss": "log_loss",
    }
    filas = []
    for anio, grupo in datos.groupby("anio_evaluacion"):
        if grupo["metrica"].duplicated().any():
            raise ValueError("Hay métricas duplicadas para el mismo año.")
        fila = {
            "fecha": fecha_scoring(),
            "modelo_version": modelo_version,
            "anio_evaluacion": int(anio),
            "umbral": float(grupo["umbral_referencia"].iloc[0]),
            "evaluacion_tipo": "temporal_exploratoria",
            "bootstrap_muestras": int(grupo["bootstrap_validos"].min()),
        }
        for registro in grupo.to_dict("records"):
            nombre = equivalencias.get(registro["metrica"])
            if nombre:
                fila[nombre] = float(registro["valor"])
                fila[f"{nombre}_ic95_inferior"] = float(registro["ic95_inferior"])
                fila[f"{nombre}_ic95_superior"] = float(registro["ic95_superior"])
        filas.append(fila)
    return pd.DataFrame(filas)


def main(
    *, engine=None, datos=None, anio_corte=None,
    incluir_metricas=True, modelo_version=VERSION_MODELO,
):
    """Puede recibir un engine aislado para demostrar el flujo sin Railway."""
    corte = resolver_corte(anio_corte=anio_corte)
    origen = load_history(corte) if datos is None else datos
    estudiantes, historico = preparar_tablas(origen, corte)
    cargar_historial(estudiantes, historico, engine=engine)

    n_metricas = 0
    ruta_metricas = PROJECT_ROOT / "reports/metricas_finales.csv"
    if incluir_metricas and ruta_metricas.is_file():
        metricas = preparar_metricas(ruta_metricas, modelo_version)
        guardar_metricas(metricas, engine=engine)
        n_metricas = len(metricas)

    resumen = {
        "anio_corte": corte,
        "estudiantes": len(estudiantes),
        "historico_anual": len(historico),
        "metricas_modelo": n_metricas,
    }
    print("Carga completada:", resumen)
    return resumen


if __name__ == "__main__":
    main()
