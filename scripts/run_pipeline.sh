#!/bin/bash
# Flujo completo local: datos -> features -> entrenamiento -> scoring
set -e
python3 scripts/cargar_datos.py
python3 -m src.models.train
python3 scripts/predict_batch.py
echo "Pipeline completado."
