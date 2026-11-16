"""API que comparte con el batch la preparación de features y la predicción."""
import logging

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from src.api.dependencies import get_db_engine, get_paquete, verificar_token
from src.api.schemas import BatchResponse, PredictRequest, PredictResponse
from src.models.predict import (
    HistorialInvalido, ModeloNoDisponible, fecha_scoring, predecir_historial,
    resolver_corte,
)

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Alerta Temprana de Abandono — Psicología",
    version="1.1.0",
    description=(
        "Prototipo con logística y recuentos. Las bandas de riesgo son "
        "provisionales; las contribuciones explicativas se expresan en log-odds."
    ),
)


@app.get("/health")
def health():
    """Comprueba el proceso sin necesitar modelo, base de datos ni Internet."""
    return {"status": "ok", "version": app.version}


@app.get("/model-info")
def model_info(paquete=Depends(get_paquete)):
    metadata = paquete["metadata"]
    return {
        "modelo": type(paquete["modelo"].named_steps["modelo"]).__name__,
        "pipeline": type(paquete["modelo"]).__name__,
        "modelo_version": metadata.get("version"),
        "n_features": len(paquete["features"]),
        "features": paquete["features"],
        "anio_corte_predeterminado": resolver_corte(paquete),
        "ventana_anios": metadata.get("target", {}).get("ventana_anios"),
        "calibracion": metadata.get("calibracion"),
        "bandas_validadas": metadata.get("bandas_riesgo_validadas", False),
        "umbral_validado": metadata.get("umbral_validado", False),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, paquete=Depends(get_paquete)):
    """Recibe la trayectoria completa, la ordena y predice el año de corte."""
    historia = pd.DataFrame([
        registro.model_dump() for registro in req.historia
    ])
    historia["id_alumno"] = req.id_alumno

    try:
        resultados = predecir_historial(historia, paquete, req.anio_corte)
    except HistorialInvalido as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModeloNoDisponible as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if resultados.empty:
        raise HTTPException(
            status_code=422,
            detail="El estudiante no tiene una observación en el año de corte.",
        )

    resultado = resultados.iloc[0].to_dict()
    # El cálculo conserva precisión completa; solo la respuesta se redondea.
    resultado["probabilidad_abandono"] = round(
        float(resultado["probabilidad_abandono"]), 6
    )
    return PredictResponse(**resultado)


@app.post(
    "/score-batch",
    response_model=BatchResponse,
    dependencies=[Depends(verificar_token)],
)
def score_batch(paquete=Depends(get_paquete), db_engine=Depends(get_db_engine)):
    """El cron invoca este endpoint; aquí no se reentrena el modelo."""
    from scripts.predict_batch import main as ejecutar_batch

    try:
        n = ejecutar_batch(engine=db_engine, paquete=paquete)
    except HistorialInvalido as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModeloNoDisponible as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("No se pudo completar el scoring batch.")
        raise HTTPException(
            status_code=503,
            detail="No se pudo completar el batch. Revisá la base y sus registros.",
        ) from exc
    return BatchResponse(registros_scorados=n, fecha=fecha_scoring().isoformat())
