"""Lectura de datos con rutas estables, incluso desde scripts o Railway."""
from pathlib import Path

import pandas as pd

# Las rutas dependen de este archivo y no del directorio de la terminal.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw(filename: str | Path = "Psicologia_2023_Anonimizado.xlsx") -> pd.DataFrame:
    """Lee el original; también acepta una ruta absoluta a otro Excel/CSV."""
    path = Path(filename)
    if not path.is_absolute():
        path = DATA_RAW / path
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_processed(filename: str | Path = "dataset_modelado.parquet") -> pd.DataFrame:
    """Lee el dataset que YA contiene las features y el target de modelado."""
    path = Path(filename)
    if not path.is_absolute():
        path = DATA_PROCESSED / path
    return pd.read_parquet(path)


def load_history(anio_corte: int | None = None) -> pd.DataFrame:
    """Recupera el historial para scoring sin excluir años por censura del target.

    Edad cero permanece como dato inválido: el pipeline entrenado la imputará.
    No se recalculan medianas con la población que se quiere predecir.
    """
    df = load_raw()
    if anio_corte is not None:
        df = df.loc[df["anio_academico"].le(anio_corte)].copy()
    return df.sort_values(["id_alumno", "anio_academico"]).reset_index(drop=True)
