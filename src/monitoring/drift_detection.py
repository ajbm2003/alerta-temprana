"""Drift con Evidently: compara datos de entrenamiento vs nuevos y persiste alertas."""
import pandas as pd
from src.db.queries import insertar_df


def detectar_drift(reference: pd.DataFrame, current: pd.DataFrame) -> int:
    """Version liviana sin dependencia pesada: PSI por variable numerica.

    Devuelve cantidad de variables con drift (PSI > 0.2) y persiste en tabla drift.
    """
    alertas = []
    cols = reference.select_dtypes("number").columns.intersection(current.columns)
    for col in cols:
        psi = _psi(reference[col], current[col])
        alertas.append({"fecha": pd.Timestamp.today(), "variable": col,
                        "metrica_drift": round(psi, 4),
                        "umbral_alerta": 0.2, "alerta": bool(psi > 0.2)})
    if alertas:
        insertar_df(pd.DataFrame(alertas), "drift")
    return sum(a["alerta"] for a in alertas)


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    breakpoints = pd.cut(expected, bins=bins, retbins=True)[1]
    e_perc = pd.cut(expected, breakpoints).value_counts(normalize=True) + 1e-4
    a_perc = pd.cut(actual, breakpoints).value_counts(normalize=True) + 1e-4
    return float(((a_perc - e_perc) * pd.Series((a_perc / e_perc).apply(__import__("math").log))).sum())
