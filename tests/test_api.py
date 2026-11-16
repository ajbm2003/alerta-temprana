"""Contrato y equivalencia real de la API; nunca se acepta un 400 como éxito."""
import copy

import pandas as pd

from src.models.predict import predecir_historial


def test_health(api_client):
    respuesta = api_client.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"


def test_info_refleja_la_logistica(api_client):
    info = api_client.get("/model-info").json()
    assert info["modelo"] == "LogisticRegression"
    assert info["n_features"] == 16
    assert info["bandas_validadas"] is False
    assert info["umbral_validado"] is False


def test_predict_coincide_con_motor(api_client, payload, paquete_sintetico):
    respuesta = api_client.post("/predict", json=payload)
    assert respuesta.status_code == 200, respuesta.text

    historial = pd.DataFrame(payload["historia"])
    historial["id_alumno"] = payload["id_alumno"]
    esperado = predecir_historial(historial, paquete_sintetico, 2021).iloc[0]
    cuerpo = respuesta.json()

    assert abs(cuerpo["probabilidad_abandono"] - esperado.probabilidad_abandono) <= 5.1e-7
    assert cuerpo["anio_base"] == 2021
    assert cuerpo["bandas_validadas"] is False
    assert len(cuerpo["factores_detalle"]) == 3
    assert len(cuerpo["factores_principales"]) == 3


def test_predict_admite_alias_anio(api_client, payload):
    for registro in payload["historia"]:
        registro["anio"] = registro.pop("anio_academico")
    assert api_client.post("/predict", json=payload).status_code == 200


def test_predict_rechaza_faltantes_en_lugar_de_inventar_ceros(api_client, payload):
    del payload["historia"][0]["inscripciones_finales"]
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predict_rechaza_historia_vacia(api_client):
    assert api_client.post(
        "/predict", json={"id_alumno": 1, "historia": []}
    ).status_code == 422


def test_predict_rechaza_anios_duplicados(api_client, payload):
    payload["historia"].append(copy.deepcopy(payload["historia"][0]))
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predict_rechaza_tasa_invalida(api_client, payload):
    payload["historia"][0]["tasa_aprobacion_asignaturas"] = 1.5
    assert api_client.post("/predict", json=payload).status_code == 422


def test_edad_cero_y_faltante_usan_la_misma_imputacion(api_client, payload):
    for registro in payload["historia"]:
        registro["edad"] = 0
    cero = api_client.post("/predict", json=payload)
    for registro in payload["historia"]:
        registro["edad"] = None
    faltante = api_client.post("/predict", json=payload)
    assert cero.status_code == faltante.status_code == 200
    assert cero.json()["probabilidad_abandono"] == faltante.json()["probabilidad_abandono"]


def test_predict_no_incorpora_futuro(api_client, payload):
    inicial = api_client.post("/predict", json=payload).json()
    futuro = copy.deepcopy(payload["historia"][0])
    futuro["anio_academico"] = 2022
    futuro["edad"] = 100
    futuro["tasa_aprobacion_asignaturas"] = 0
    payload["historia"].append(futuro)
    posterior = api_client.post("/predict", json=payload)
    assert posterior.status_code == 200
    assert posterior.json()["probabilidad_abandono"] == inicial["probabilidad_abandono"]


def test_batch_exige_token(api_client):
    assert api_client.post("/score-batch").status_code == 401
    assert api_client.post(
        "/score-batch", headers={"Authorization": "Bearer incorrecto"}
    ).status_code == 401
