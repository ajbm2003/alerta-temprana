"""Reproduce el pipeline final del notebook 11; no reconstruye el target."""
import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import PROJECT_ROOT, load_processed, load_raw
from src.features.build_features import FEATURES_MODELO_FINAL

MODELS_DIR = PROJECT_ROOT / "models"
VERSION_MODELO = "logistica-recuentos-v1"


def crear_pipeline() -> Pipeline:
    """La imputación y el escalado viajan con el estimador al guardarlo."""
    return Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
        ("modelo", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, class_weight=None,
        )),
    ])


def preparar_modelado(df: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Conserva las etiquetas y recupera la marca de las edades imputadas."""
    claves = ["id_alumno", "anio_academico"]
    if df.duplicated(claves).any() or original.duplicated(claves).any():
        raise ValueError("Hay registros estudiante-año duplicados.")
    faltantes = set(FEATURES_MODELO_FINAL + ["abandono"] + claves) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de modelado: {sorted(faltantes)}")
    if not df["abandono"].isin([0, 1]).all():
        raise ValueError("El target debe estar preparado y contener solo 0 y 1.")

    edades = original[claves + ["edad"]].rename(columns={"edad": "edad_original"})
    preparado = (
        df.sort_values(claves).reset_index(drop=True)
        .merge(edades, on=claves, how="left", sort=False, validate="one_to_one")
    )
    if preparado["edad_original"].isna().any():
        raise ValueError("No se encontraron las edades originales de todos los registros.")

    # Evita usar la mediana por cohorte calculada antes de la validación.
    preparado["edad"] = preparado["edad"].astype(float)
    preparado.loc[preparado["edad_original"].eq(0), "edad"] = np.nan
    return preparado


def entrenar(ruta_modelo: str | Path | None = None) -> dict:
    """Ajusta todos los datos etiquetados y guarda el paquete reproducible.

    Las métricas se obtienen de la evaluación temporal ya realizada, nunca
    de los aciertos sobre este entrenamiento completo.
    """
    with open(PROJECT_ROOT / "configs/config.yaml", encoding="utf-8") as archivo:
        cfg = yaml.safe_load(archivo)

    df = preparar_modelado(load_processed(), load_raw())
    cierre = int(cfg["target"]["censurar_hasta"])
    ventana = int(cfg["target"]["ventana_anios"])
    if not (df["anio_academico"] + ventana).le(cierre).all():
        raise ValueError("El parquet contiene etiquetas sin seguimiento completo.")

    X = df[FEATURES_MODELO_FINAL].copy()
    y = df["abandono"].astype(int)
    if y.nunique() != 2 or np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Se necesitan ambas clases y predictores sin infinitos.")

    modelo = crear_pipeline()
    modelo.fit(X, y)

    # El fondo transformado permite calcular después SHAP lineal en log-odds.
    fondo = np.mean(modelo[:-1].transform(X), axis=0)
    estimador = modelo.named_steps["modelo"]
    base = float(estimador.intercept_[0] + fondo @ estimador.coef_[0])
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

    metadata = {
        "version": VERSION_MODELO,
        "modelo": "logistica_recuentos",
        "estado": "prototipo_evaluado_exploratoriamente",
        "features": list(FEATURES_MODELO_FINAL),
        "calibracion": "sin_recalibracion_adicional",
        "umbral_referencia": 0.5,
        "umbral_validado": False,
        "bandas_riesgo_validadas": False,
        "target": {"ventana_anios": ventana, "cierre_seguimiento": cierre},
        "entrenamiento": {
            "registros": len(df),
            "estudiantes": int(df["id_alumno"].nunique()),
            "positivos": int(y.sum()),
            "primer_anio": int(df["anio_academico"].min()),
            "ultimo_anio": int(df["anio_academico"].max()),
        },
        "evaluacion": {
            "tipo": "temporal_y_por_estudiante_exploratoria",
            "archivo_metricas": "reports/metricas_finales.csv",
            "metricas_sobre_entrenamiento": False,
        },
        "explicabilidad": {
            "metodo": "shap_lineal_intervencional_analitico",
            "escala": "log_odds",
            "supuesto": "fondo_con_variables_independientes",
            "media_fondo_transformado": fondo.tolist(),
            "valor_base_log_odds": base,
        },
        "huellas": {
            "dataset_modelado_sha256": sha(
                PROJECT_ROOT / "data/processed/dataset_modelado.parquet"
            ),
            "config_sha256": sha(PROJECT_ROOT / "configs/config.yaml"),
        },
        "versiones": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }

    ruta = Path(ruta_modelo) if ruta_modelo else MODELS_DIR / "modelo_final.joblib"
    if not ruta.is_absolute():
        ruta = PROJECT_ROOT / ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "modelo": modelo,
        "features": list(FEATURES_MODELO_FINAL),
        "metadata": metadata,
    }, ruta)

    resumen = {
        "modelo": metadata["modelo"],
        "version": VERSION_MODELO,
        "registros": len(df),
        "estudiantes": metadata["entrenamiento"]["estudiantes"],
        "n_features": len(FEATURES_MODELO_FINAL),
        "ruta": str(ruta),
    }
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return resumen


def publicar_en_hf(ruta_modelo: str | Path | None = None) -> None:
    """Publicación explícita: entrenar o importar este módulo NO sube archivos."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    from huggingface_hub import HfApi

    ruta = Path(ruta_modelo) if ruta_modelo else MODELS_DIR / "modelo_final.joblib"
    if not ruta.is_absolute():
        ruta = PROJECT_ROOT / ruta
    repo = os.environ.get("HF_MODEL_REPO")
    if not repo:
        raise ValueError("Configurá HF_MODEL_REPO antes de publicar.")
    HfApi().upload_file(
        path_or_fileobj=str(ruta),
        path_in_repo="modelo_final.joblib",
        repo_id=repo,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
    )
    print("Modelo publicado en el repositorio configurado.")


if __name__ == "__main__":
    entrenar()
