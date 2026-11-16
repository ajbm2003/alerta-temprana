PY = python3
PIP = pip3

install:
	$(PY) -m pip install -r requirements.txt

db-up:
	docker compose up -d db

db-down:
	docker compose down

pipeline:
	$(PY) scripts/cargar_datos.py
	$(PY) -m src.models.train
	$(PY) scripts/predict_batch.py

api:
	$(PY) -m uvicorn src.api.main:app --reload

test:
	$(PY) -m pytest tests/ -v

demo:
	$(PY) scripts/demo_local.py --guardar-ejemplo reports/ejemplo_predict.json --guardar-resultado reports/demo_integracion.json

.PHONY: install db-up db-down pipeline api test demo
