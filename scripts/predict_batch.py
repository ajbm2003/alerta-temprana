"""El cron puntúa el año de corte; no entrena ni reconstruye el target."""
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.db.queries import guardar_predicciones, leer
from src.models.predict import (
    cargar_paquete, fecha_scoring, predecir_historial, resolver_corte,
)


def main(*, engine=None, paquete=None, anio_corte=None, fecha=None) -> int:
    """Guarda una fila por estudiante activo en el corte y por día/modelo."""
    paquete = cargar_paquete() if paquete is None else paquete
    corte = resolver_corte(paquete, anio_corte)
    historico = leer("historico_anual", engine=engine)

    # Esta función es exactamente la que utiliza POST /predict.
    resultados = predecir_historial(historico, paquete, corte)
    if resultados.empty:
        print(f"No hay observaciones para puntuar en {corte}.")
        return 0

    columnas = [
        "id_alumno", "anio_base", "probabilidad_abandono", "banda_riesgo",
        "modelo_version", "bandas_validadas", "umbral_referencia",
        "valor_base_log_odds",
    ]
    salida = resultados[columnas].copy()
    salida["fecha_scoring"] = fecha or fecha_scoring()
    salida["explicaciones_json"] = resultados["factores_detalle"].map(
        lambda factores: json.dumps(factores, ensure_ascii=False)
    )

    for posicion in range(3):
        salida[f"factor_{posicion + 1}"] = resultados["factores_principales"].map(
            lambda factores, i=posicion: factores[i] if len(factores) > i else None
        )

    # ON CONFLICT actualiza si el cron se repite en el mismo día.
    guardar_predicciones(salida, engine=engine)
    print(f"Scoring completado: {len(salida)} estudiantes, año {corte}.")
    return len(salida)


if __name__ == "__main__":
    main()
