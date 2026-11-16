# Sistema de Alerta Temprana de Abandono
## Licenciatura en Psicología — Práctica Profesional Supervisada

El modelo elegido es una regresión logística con dieciséis predictores:
las trece features originales y tres recuentos académicos. Su paquete
incluye imputación y escalado. La API y el batch reconstruyen las mismas
features a partir del historial y utilizan ese mismo pipeline.

Los notebooks conservan el análisis exploratorio y la evaluación temporal.
El despliegue de Railway todavía debe configurarse; este repositorio no
afirma que la aplicación ya esté publicada.

## Preparación

Desde la raíz del proyecto, usar Python 3.13 y un entorno virtual:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

El entrenamiento necesita el Excel anonimizado en data/raw/ y el parquet
preparado por los notebooks en data/processed/dataset_modelado.parquet.
No se vuelven a construir etiquetas sobre ese parquet censurado.

## Demostración local completa

Con los artefactos del notebook 11 ya generados:

    .venv/bin/python scripts/demo_local.py --guardar-ejemplo reports/ejemplo_predict.json --guardar-resultado reports/demo_integracion.json

La demo crea una base SQLite temporal, carga el historial completo hasta
2025 y las métricas reales, ejecuta el batch, llama a la API con TestClient
y compara las probabilidades con reports/scoring_demostracion.csv.
También comprueba que repetir el batch no duplica registros y que el
endpoint de batch rechaza una petición sin token.

No utiliza DATABASE_URL, no consulta Railway ni publica el modelo.
La base temporal se elimina al terminar. El archivo demo_integracion.json
contiene el resultado comprobado y ejemplo_predict.json contiene una
trayectoria real anonimizada para repetir POST /predict.

## API en el navegador

    MODEL_PATH=models/modelo_final.joblib .venv/bin/python -m uvicorn src.api.main:app --reload

Abrir http://127.0.0.1:8000/docs. También se puede enviar el ejemplo:

    curl --fail-with-body -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @reports/ejemplo_predict.json

/predict recibe id_alumno, anio_corte y la historia anual completa.
Acepta anio como alias de anio_academico. Ordena la historia, excluye años
posteriores al corte y calcula delta, brecha, ritmo y racha sin finales.
Solo puntúa estudiantes con observación en el año de corte.

Los datos académicos necesarios son obligatorios: no se inventan ceros.
Edad cero, ausente o nula se imputa con la mediana aprendida por el pipeline.
La respuesta incluye versión, tres factores explicativos y sus aportes
en log-odds. Esos aportes describen el modelo y no son efectos causales.

Las bandas bajo/medio/alto conservan provisionalmente los cortes 0,30 y
0,60. La respuesta declara bandas_validadas=false; no equivalen a criterios
de atención aprobados. El corte binario 0,50 también sigue siendo referencia.

## PostgreSQL y batch

Copiar .env.example a .env y completar la conexión propia, sin subir
secretos al repositorio. DATABASE_URL tiene prioridad sobre DB_HOST y
las otras variables individuales.

    python scripts/cargar_datos.py
    python scripts/predict_batch.py

La carga crea las cinco tablas y actualiza estudiantes e historial mediante
upsert, sin reemplazar tablas ni eliminar claves foráneas. Incluye
inscripciones_finales y todas las observaciones disponibles hasta el corte,
también las recientes que no tienen target. Carga las métricas de bootstrap
en metricas_modelo. drift se crea vacío hasta registrar una comparación
real: no se fabrican alertas para rellenarlo.

Una predicción se identifica por estudiante, año base, versión y fecha.
Repetir el batch el mismo día actualiza esa fila; otro día conserva historia.
Las tablas anteriores reciben las columnas nuevas al inicializarse.
Si existen duplicados antiguos que impidan el índice único, la carga falla
sin borrarlos: hay que revisarlos antes de continuar.

SCORING_YEAR indica el año académico completo que se quiere puntuar.
Por defecto se utiliza el cierre conocido del paquete/configuración, 2025.
Cambiarlo para datos nuevos completos no requiere ejecutar train.py cada día.

## Cron mediante la API

Configurar API_TOKEN y enviar Authorization: Bearer con ese token a
POST /score-batch. Si API_TOKEN no existe, el endpoint de escritura queda
deshabilitado. /predict y /health no requieren una base para funcionar.

El workflow .github/workflows/score.yml necesita los secretos API_URL y
API_TOKEN. Un HTTP de error hace fallar el workflow. El cron solo predice:
no reentrena. Power BI requiere además su propio refresh para mostrar las
nuevas filas de PostgreSQL.

## Reproducir el entrenamiento y publicar

    python -m src.models.train

Ese comando reproduce la logística de los notebooks 09–11 con las dieciséis
entradas, recupera la marca de edades inválidas del original y guarda
models/modelo_final.joblib. No calcula métricas de entrenamiento como si
fueran validación. El paquete conserva versiones, huellas y fondo de
explicación. Actualizar VERSION_MODELO al publicar una nueva versión.

La publicación es una acción separada y explícita:

    python -c "from src.models.train import publicar_en_hf; publicar_en_hf()"

Configurar HF_MODEL_REPO y HF_TOKEN para publicar o descargar un repositorio
privado. MODEL_PATH tiene prioridad para usar un archivo local.

## Pruebas

Sin PostgreSQL local:

    SKIP_DB_TESTS=1 .venv/bin/python -m pytest tests/ -v

Se prueban contrato de API, separación temporal, imputación, consistencia de
features, migración sin pérdida de filas e idempotencia del batch con SQLite.
Con los artefactos locales, también se comparan los 663 casos del notebook
y se reproduce el entrenamiento. Esas pruebas se omiten en CI cuando los
datos/modelos excluidos por .gitignore no están disponibles.

La demo SQLite verifica el recorrido local; la conexión PostgreSQL de
Railway se debe comprobar al configurar el despliegue real.
También se verificaron carga y scoring sobre PostgreSQL 15 local temporal,
incluida la repetición sin duplicados; la suite completa pasó sus 19 pruebas.

## Arquitectura y estado

Mac: notebooks, preparación del dataset, entrenamiento y carga inicial.
GitHub: código y CI. Hugging Face: paquete versionado.
Railway: API, PostgreSQL y ejecución programada.
Power BI: lectura de predicciones, métricas y drift desde PostgreSQL.

railway.toml inicia Uvicorn, no train.py. El contenedor incluye scripts/
para que /score-batch funcione. Configurar DATABASE_URL, HF_MODEL_REPO,
HF_TOKEN, API_TOKEN y SCORING_YEAR en Railway; no subir .env ni datasets.

La evaluación es retrospectiva y exploratoria: 2022 y 2023 ya fueron
inspeccionados. Los intervalos bootstrap mantienen fijas las predicciones y
no incluyen toda la incertidumbre de entrenamiento o selección. La calidad
operativa y las bandas deben seguir documentándose en reports/memoria.md.
