"""Preparación y predicción únicas para notebook, API y batch."""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from scipy.special import expit

from src.data.load_data import PROJECT_ROOT
from src.features.build_features import (
    FEATURES_COLS, FEATURES_MODELO_FINAL, RECUENTOS_COLS, agregar_features,
)

# Son las entradas históricas necesarias para construir las cuatro derivadas.
CAMPOS_HISTORIAL = list(dict.fromkeys(
    ["id_alumno", "anio_academico"] + FEATURES_COLS[:9] + RECUENTOS_COLS
))
COLUMNAS_RESULTADO = [
    "id_alumno", "anio_base", "probabilidad_abandono", "banda_riesgo",
    "modelo_version", "bandas_validadas", "umbral_referencia",
    "factores_principales", "factores_detalle", "valor_base_log_odds",
]


class ModeloNoDisponible(RuntimeError):
    """El paquete no se encuentra o no corresponde al modelo elegido."""


class HistorialInvalido(ValueError):
    """Faltan datos o el historial no permite preparar una predicción válida."""


def fecha_scoring():
    """Una misma fecha para API y cron, en Guayaquil (UTC−5)."""
    return datetime.now(timezone(timedelta(hours=-5))).date()


def validar_paquete(paquete: dict) -> dict:
    """Impide servir por error un LightGBM anterior o un esquema diferente."""
    try:
        modelo = paquete["modelo"]
        features = list(paquete["features"])
        metadata = paquete["metadata"]
        estimador = modelo.named_steps["modelo"]
        fondo = metadata["explicabilidad"]["media_fondo_transformado"]
        if features != FEATURES_MODELO_FINAL:
            raise ValueError("Las entradas no coinciden con los dieciséis predictores finales.")
        if len(fondo) != len(features) or not np.isfinite(fondo).all():
            raise ValueError("El fondo de explicación no es válido.")
        if not hasattr(estimador, "coef_") or estimador.coef_.shape != (1, len(features)):
            raise ValueError("El estimador no es una logística binaria compatible.")
        if list(modelo.feature_names_in_) != features:
            raise ValueError("El orden de las columnas no coincide con el pipeline.")
        if list(modelo.classes_) != [0, 1]:
            raise ValueError("El paquete no usa abandono=1.")
        if metadata.get("features", features) != features:
            raise ValueError("Los metadatos tienen otro esquema de entrada.")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ModeloNoDisponible(f"Paquete de modelo inválido: {exc}") from exc
    return paquete


def cargar_paquete(ruta_modelo: str | Path | None = None) -> dict:
    """Carga local explícita, MODEL_PATH, HF configurado o archivo local.

    Hugging Face se importa solo cuando se necesita una descarga. Una ruta
    explícita permite probar localmente sin conectarse a ningún servicio.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        ruta = ruta_modelo or os.environ.get("MODEL_PATH")
        if ruta:
            ruta = Path(ruta)
            if not ruta.is_absolute():
                ruta = PROJECT_ROOT / ruta
        elif os.environ.get("HF_MODEL_REPO"):
            from huggingface_hub import hf_hub_download

            ruta = hf_hub_download(
                # Al pegar variables en Railway pueden quedar espacios o saltos finales.
                repo_id=os.environ["HF_MODEL_REPO"].strip(),
                filename="modelo_final.joblib",
                repo_type="model",
                token=(os.environ.get("HF_TOKEN") or "").strip() or None,
            )
        else:
            ruta = PROJECT_ROOT / "models/modelo_final.joblib"

        # Solo se deben cargar paquetes propios o de repositorios de confianza.
        return validar_paquete(joblib.load(ruta))
    except ModeloNoDisponible:
        raise
    except Exception as exc:
        raise ModeloNoDisponible(
            "No se pudo cargar el modelo. Revisá MODEL_PATH o el repositorio configurado."
        ) from exc


def cargar_modelo():
    """Conserva el contrato anterior para otros módulos del proyecto."""
    paquete = cargar_paquete()
    return paquete["modelo"], paquete["features"]


def resolver_corte(paquete: dict | None = None, anio_corte: int | None = None) -> int:
    """El corte de scoring es explícito y no depende del año del reloj.

    SCORING_YEAR permite avanzar a otro año completo sin reentrenar por cron.
    Mientras no se configure, se usa el cierre conocido del paquete/config.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    corte = anio_corte if anio_corte is not None else os.environ.get("SCORING_YEAR")
    if corte is None and paquete is not None:
        corte = paquete.get("metadata", {}).get("target", {}).get("cierre_seguimiento")
    if corte is None:
        with open(PROJECT_ROOT / "configs/config.yaml", encoding="utf-8") as archivo:
            corte = yaml.safe_load(archivo)["target"]["censurar_hasta"]
    try:
        entero = int(corte)
        if float(corte) != entero or not 1900 <= entero <= 2100:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise HistorialInvalido("El año de corte debe ser un entero entre 1900 y 2100.") from exc
    return entero


def preparar_historial(historial: pd.DataFrame, anio_corte: int) -> pd.DataFrame:
    """Normaliza, recorta antes de derivar features y conserva edad faltante."""
    df = historial.copy()
    # SQLAlchemy puede devolver nombres quoted_name; sklearn exige str puros.
    df.columns = [str(columna) for columna in df.columns]
    if "anio" in df.columns:
        if "anio_academico" in df.columns:
            if not df["anio"].equals(df["anio_academico"]):
                raise HistorialInvalido("anio y anio_academico contienen valores distintos.")
            df = df.drop(columns="anio")
        else:
            df = df.rename(columns={"anio": "anio_academico"})

    if "edad" not in df:
        df["edad"] = np.nan
    faltantes = set(CAMPOS_HISTORIAL) - set(df.columns)
    if faltantes:
        raise HistorialInvalido(f"Faltan campos históricos: {', '.join(sorted(faltantes))}")
    try:
        df = df[CAMPOS_HISTORIAL].copy()
        for col in CAMPOS_HISTORIAL:
            df[col] = pd.to_numeric(df[col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise HistorialInvalido("Los campos históricos deben ser numéricos.") from exc

    # Se quita el futuro ANTES de diff y rachas para reproducir el notebook.
    df = df.loc[df["anio_academico"].le(anio_corte)].copy()
    if df.empty:
        return df

    obligatorias = [c for c in CAMPOS_HISTORIAL if c != "edad"]
    if not np.isfinite(df[obligatorias].to_numpy(dtype=float)).all():
        raise HistorialInvalido("Hay valores faltantes o infinitos fuera de edad.")

    for col in ["id_alumno", "anio_academico"]:
        if not df[col].eq(np.floor(df[col])).all():
            raise HistorialInvalido(f"{col} debe contener enteros.")
        df[col] = df[col].astype(int)
    if not df["id_alumno"].gt(0).all() or not df["anio_academico"].between(1900, 2100).all():
        raise HistorialInvalido("Identificador o año inválido.")
    if df.duplicated(["id_alumno", "anio_academico"]).any():
        raise HistorialInvalido("Hay más de una observación del mismo estudiante y año.")

    tasas = [
        "tasa_aprobacion_asignaturas", "tasa_desaprobacion_asignaturas",
        "tasa_ausencias_finales", "porcentaje_avance_carrera", "porcentaje_avance_cursado",
    ]
    if not df[tasas].ge(0).all().all() or not df[tasas].le(1).all().all():
        raise HistorialInvalido("Las tasas y avances deben estar entre 0 y 1.")
    conteos = RECUENTOS_COLS + ["cantidad_pausas", "maxima_duracion_pausa", "antiguedad_carrera"]
    if not df[conteos].ge(0).all().all():
        raise HistorialInvalido("Los recuentos y la antigüedad no pueden ser negativos.")
    if not df[conteos].eq(np.floor(df[conteos])).all().all():
        raise HistorialInvalido("Los recuentos y la antigüedad deben ser enteros.")
    if df["finales_aprobados"].gt(df["inscripciones_finales"]).any():
        raise HistorialInvalido("Los finales aprobados no pueden superar las inscripciones.")
    if not df["edad"].dropna().between(0, 130).all():
        raise HistorialInvalido("La edad debe estar entre 0 y 130, o faltar para imputarse.")

    df = df.sort_values(["id_alumno", "anio_academico"]).reset_index(drop=True)
    df["edad"] = df["edad"].fillna(0)
    features = agregar_features(df)

    # agregar_features conserva la convención histórica de rellenar ceros.
    # Restauramos edad inválida a NaN para que impute el pipeline ya ajustado.
    features["edad"] = features["edad"].astype(float)
    features.loc[features["edad"].eq(0), "edad"] = np.nan
    return features


def banda(prob: float) -> str:
    """Bandas provisionales existentes, compartidas por API y batch."""
    if prob >= 0.60:
        return "alto"
    if prob >= 0.30:
        return "medio"
    return "bajo"


def explicar_registros(paquete: dict, X: pd.DataFrame):
    """SHAP lineal analítico: misma fórmula y fondo que en el notebook 11."""
    modelo = paquete["modelo"]
    Z = modelo[:-1].transform(X)
    valores = modelo.named_steps["imputacion"].transform(X)
    estimador = modelo.named_steps["modelo"]
    fondo = np.asarray(
        paquete["metadata"]["explicabilidad"]["media_fondo_transformado"], dtype=float
    )
    coef = estimador.coef_[0]
    contribuciones = (Z - fondo) * coef
    base = float(estimador.intercept_[0] + fondo @ coef)
    margen = base + contribuciones.sum(axis=1)

    if not np.allclose(expit(margen), modelo.predict_proba(X)[:, 1], atol=1e-10):
        raise ModeloNoDisponible("La explicación no reconstruye la probabilidad del paquete.")

    explicaciones = []
    for fila in range(len(X)):
        principales = np.argsort(-np.abs(contribuciones[fila]), kind="stable")[:3]
        detalle = []
        for j in principales:
            aporte = float(contribuciones[fila, j])
            detalle.append({
                "feature": paquete["features"][j],
                "valor_usado": float(valores[fila, j]),
                "contribucion_log_odds": aporte,
                "efecto": "aumenta" if aporte > 0 else "reduce" if aporte < 0 else "neutro",
            })
        explicaciones.append(detalle)
    return explicaciones, base


def predecir_historial(
    historial: pd.DataFrame,
    paquete: dict | None = None,
    anio_corte: int | None = None,
) -> pd.DataFrame:
    """Calcula una predicción por estudiante con observación en el año de corte."""
    paquete = validar_paquete(paquete) if paquete is not None else cargar_paquete()
    corte = resolver_corte(paquete, anio_corte)
    features = preparar_historial(historial, corte)
    if features.empty:
        return pd.DataFrame(columns=COLUMNAS_RESULTADO)

    actuales = features.loc[features["anio_academico"].eq(corte)].copy()
    if actuales.empty:
        return pd.DataFrame(columns=COLUMNAS_RESULTADO)

    X = actuales[paquete["features"]]
    prob = paquete["modelo"].predict_proba(X)[:, 1]
    detalles, base = explicar_registros(paquete, X)
    metadata = paquete["metadata"]

    resultado = actuales[["id_alumno", "anio_academico"]].rename(
        columns={"anio_academico": "anio_base"}
    ).reset_index(drop=True)
    resultado["probabilidad_abandono"] = prob
    resultado["banda_riesgo"] = [banda(float(p)) for p in prob]
    resultado["modelo_version"] = metadata.get("version", "sin-version")
    resultado["bandas_validadas"] = bool(metadata.get("bandas_riesgo_validadas", False))
    resultado["umbral_referencia"] = float(metadata.get("umbral_referencia", 0.5))
    resultado["factores_detalle"] = detalles
    resultado["factores_principales"] = [
        [
            f"{factor['feature']}: "
            f"{factor['efecto'] if factor['efecto'] != 'neutro' else 'no cambia'} "
            "el riesgo estimado"
            for factor in fila
        ]
        for fila in detalles
    ]
    resultado["valor_base_log_odds"] = base
    return resultado[COLUMNAS_RESULTADO]


def predecir(modelo, features, registro: dict) -> float:
    """Compatibilidad para entradas que YA contienen todas las features."""
    faltantes = set(features) - set(registro)
    if faltantes:
        raise HistorialInvalido(f"Faltan features: {sorted(faltantes)}")
    return float(modelo.predict_proba(pd.DataFrame([registro])[features])[0, 1])
