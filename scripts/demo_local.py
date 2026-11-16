"""Demuestra el flujo real con una base temporal; nunca usa DATABASE_URL."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from scripts.cargar_datos import main as cargar_datos
from scripts.predict_batch import main as predict_batch
from src.api.dependencies import get_db_engine, get_paquete
from src.api.main import app
from src.api.schemas import HistoriaAnual
from src.data.load_data import PROJECT_ROOT, load_raw
from src.db.connection import crear_engine
from src.db.queries import leer
from src.models.predict import cargar_paquete, resolver_corte


def construir_payload(datos, id_alumno, anio_corte):
    """Selecciona los campos del contrato; se envía la trayectoria, no features."""
    campos = list(HistoriaAnual.model_fields)
    historia = datos.loc[datos["id_alumno"].eq(id_alumno), campos]
    # Se invierte para demostrar que el servidor ordena los años correctamente.
    registros = historia.iloc[::-1].to_dict("records")
    return {
        "id_alumno": int(id_alumno),
        "anio_corte": int(anio_corte),
        "historia": registros,
    }


def main(guardar_ejemplo=None, guardar_resultado=None):
    paquete = cargar_paquete(PROJECT_ROOT / "models/modelo_final.joblib")
    corte = resolver_corte(paquete, paquete["metadata"]["target"]["cierre_seguimiento"])
    datos = load_raw()
    overrides_previos = app.dependency_overrides.copy()
    entorno_previo = {k: os.environ.get(k) for k in ["API_TOKEN", "SCORING_YEAR"]}

    # El token es de demostración y solo existe dentro de este proceso.
    token_demo = "demo-local-sin-credenciales-reales"
    os.environ["API_TOKEN"] = token_demo
    os.environ["SCORING_YEAR"] = str(corte)

    try:
        with tempfile.TemporaryDirectory(prefix="alerta-demo-") as temporal:
            engine = crear_engine(f"sqlite:///{Path(temporal) / 'demo.sqlite'}")
            try:
                carga = cargar_datos(
                    engine=engine, datos=datos, anio_corte=corte,
                    modelo_version=paquete["metadata"]["version"],
                )
                procesados = predict_batch(engine=engine, paquete=paquete, anio_corte=corte)
                predicciones = leer("predicciones", engine=engine)

                # Compara todo el lote con la demostración ya exportada del notebook.
                referencia = pd.read_csv(PROJECT_ROOT / "reports/scoring_demostracion.csv")
                comparacion = predicciones.merge(
                    referencia[["id_alumno", "anio_academico", "probabilidad_abandono"]],
                    left_on=["id_alumno", "anio_base"],
                    right_on=["id_alumno", "anio_academico"],
                    suffixes=("_batch", "_notebook"),
                    validate="one_to_one",
                )
                assert len(comparacion) == procesados == len(referencia)
                diferencia = (
                    comparacion["probabilidad_abandono_batch"]
                    - comparacion["probabilidad_abandono_notebook"]
                ).abs()
                np.testing.assert_allclose(
                    comparacion["probabilidad_abandono_batch"],
                    comparacion["probabilidad_abandono_notebook"],
                    atol=1e-12, rtol=1e-12,
                )

                elegido = predicciones.sort_values(
                    "probabilidad_abandono", ascending=False
                ).iloc[0]
                payload = construir_payload(datos, int(elegido["id_alumno"]), corte)

                # Solo se cambian las dependencias de esta demo, no el código de la API.
                app.dependency_overrides[get_paquete] = lambda: paquete
                app.dependency_overrides[get_db_engine] = lambda: engine

                with TestClient(app) as client:
                    respuesta = client.post("/predict", json=payload)
                    assert respuesta.status_code == 200, respuesta.text
                    cuerpo = respuesta.json()
                    assert abs(
                        cuerpo["probabilidad_abandono"]
                        - elegido["probabilidad_abandono"]
                    ) <= 5.1e-7

                    sin_token = client.post("/score-batch")
                    assert sin_token.status_code == 401

                    repeticion = client.post(
                        "/score-batch",
                        headers={"Authorization": f"Bearer {token_demo}"},
                    )
                    assert repeticion.status_code == 200, repeticion.text

                finales = leer("predicciones", engine=engine)
                assert len(finales) == procesados  # Reintentar no duplica filas.

                tablas = {
                    nombre: len(leer(nombre, engine=engine))
                    for nombre in [
                        "estudiantes", "historico_anual", "predicciones",
                        "metricas_modelo", "drift",
                    ]
                }
                resumen = {
                    "modo": "demo_local_base_temporal",
                    "modelo_version": paquete["metadata"]["version"],
                    "anio_corte": corte,
                    "carga": carga,
                    "filas_tablas": tablas,
                    "registros_comparados_con_notebook": len(comparacion),
                    "diferencia_maxima_batch_notebook": float(diferencia.max()),
                    "repeticion_batch_sin_duplicados": True,
                    "batch_sin_token_http": sin_token.status_code,
                    "predict_http": respuesta.status_code,
                    "prediccion_api": cuerpo,
                }

                if guardar_ejemplo:
                    ruta = Path(guardar_ejemplo)
                    ruta.parent.mkdir(parents=True, exist_ok=True)
                    ruta.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                if guardar_resultado:
                    ruta = Path(guardar_resultado)
                    ruta.parent.mkdir(parents=True, exist_ok=True)
                    ruta.write_text(
                        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )

                print(json.dumps(resumen, ensure_ascii=False, indent=2))
                return resumen
            finally:
                engine.dispose()
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(overrides_previos)
        for clave, valor in entorno_previo.items():
            if valor is None:
                os.environ.pop(clave, None)
            else:
                os.environ[clave] = valor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guardar-ejemplo", help="Ruta para guardar un JSON de POST /predict.")
    parser.add_argument("--guardar-resultado", help="Ruta para guardar la verificación de la demo.")
    args = parser.parse_args()
    main(args.guardar_ejemplo, args.guardar_resultado)
