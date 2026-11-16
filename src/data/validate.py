"""Validaciones de calidad sobre el dataset crudo."""
import pandas as pd

ASIGNATURAS_PLAN = 48
FINALES_PLAN = 44


def validar(df: pd.DataFrame) -> list[str]:
    """Devuelve lista de problemas encontrados (vacia si todo ok)."""
    problemas = []
    if df.isna().any().any():
        problemas.append("Existen valores nulos.")
    if (df["edad"] == 0).any():
        n = int((df["edad"] == 0).sum())
        problemas.append(f"edad=0 en {n} registros: imputar o excluir.")
    if (df["asignaturas_aprobadas"] > df["asignaturas_cursadas"]).any():
        problemas.append("asignaturas_aprobadas > cursadas en algun registro.")
    if df["asignaturas_aprobadas_acumuladas"].max() > ASIGNATURAS_PLAN:
        problemas.append("Acumulado de asignaturas supera el plan (48).")
    if df["finales_aprobados_acumulados"].max() > FINALES_PLAN:
        problemas.append("Acumulado de finales supera el plan (44).")
    return problemas


def corregir_edad(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa edad=0 con la mediana por cohorte de carrera."""
    df = df.copy()
    mediana = df.loc[df["edad"] > 0].groupby("cohorte_carrera")["edad"].median()
    mask = df["edad"] == 0
    df.loc[mask, "edad"] = df.loc[mask, "cohorte_carrera"].map(mediana)
    df["edad"] = df["edad"].fillna(df.loc[df["edad"] > 0, "edad"].median())
    return df
