"""Las cinco tablas del proyecto, sin borrar datos al inicializarlas."""
from sqlalchemy import (
    Boolean, Column, Date, Float, ForeignKey, Index, Integer,
    MetaData, String, Table, Text, inspect, text,
)

metadata = MetaData()

estudiantes = Table(
    "estudiantes", metadata,
    Column("id_alumno", Integer, primary_key=True),
    Column("cohorte_carrera", Integer),
    Column("cohorte_universidad", Integer),
    Column("tuvo_actividad_previa", Integer),
    Column("anios_actividad_previa", Integer),
)

historico = Table(
    "historico_anual", metadata,
    Column("id_alumno", Integer, ForeignKey("estudiantes.id_alumno"), primary_key=True),
    Column("anio_academico", Integer, primary_key=True),
    Column("edad", Float),
    Column("asignaturas_cursadas", Integer),
    Column("asignaturas_aprobadas", Integer),
    Column("inscripciones_finales", Integer),
    Column("tasa_aprobacion_asignaturas", Float),
    Column("tasa_desaprobacion_asignaturas", Float),
    Column("finales_aprobados", Integer),
    Column("finales_ausentes", Integer),
    Column("tasa_ausencias_finales", Float),
    Column("equivalencias_asignaturas", Integer),
    Column("porcentaje_avance_cursado", Float),
    Column("porcentaje_avance_carrera", Float),
    Column("cantidad_pausas", Integer),
    Column("maxima_duracion_pausa", Integer),
    Column("antiguedad_carrera", Integer),
    Column("cambio_plan", Integer),
)

predicciones = Table(
    "predicciones", metadata,
    Column("id_prediccion", Integer, primary_key=True, autoincrement=True),
    Column("id_alumno", Integer, ForeignKey("estudiantes.id_alumno"), nullable=False),
    Column("anio_base", Integer, nullable=False),
    Column("probabilidad_abandono", Float),
    Column("banda_riesgo", String(10)),
    Column("factor_1", Text),
    Column("factor_2", Text),
    Column("factor_3", Text),
    Column("modelo_version", String(80), nullable=False),
    Column("fecha_scoring", Date, nullable=False),
    Column("bandas_validadas", Boolean),
    Column("umbral_referencia", Float),
    Column("explicaciones_json", Text),
    Column("valor_base_log_odds", Float),
)

# Una repetición del mismo día actualiza la fila, en lugar de duplicarla.
indice_predicciones = Index(
    "uq_predicciones_scoring",
    predicciones.c.id_alumno,
    predicciones.c.anio_base,
    predicciones.c.modelo_version,
    predicciones.c.fecha_scoring,
    unique=True,
)

metricas = Table(
    "metricas_modelo", metadata,
    Column("id_metrica", Integer, primary_key=True, autoincrement=True),
    Column("fecha", Date),
    Column("modelo_version", String(80)),
    Column("anio_evaluacion", Integer),
    Column("umbral", Float),
    Column("evaluacion_tipo", String(80)),
    Column("bootstrap_muestras", Integer),
    *[
        Column(nombre, Float)
        for metrica in ["roc_auc", "pr_auc", "recall", "precision", "brier", "log_loss"]
        for nombre in [metrica, f"{metrica}_ic95_inferior", f"{metrica}_ic95_superior"]
    ],
)
indice_metricas = Index(
    "uq_metricas_modelo_evaluacion",
    metricas.c.modelo_version,
    metricas.c.anio_evaluacion,
    unique=True,
)

drift = Table(
    "drift", metadata,
    Column("id_drift", Integer, primary_key=True, autoincrement=True),
    Column("fecha", Date),
    Column("variable", String(100)),
    Column("metrica_drift", Float),
    Column("umbral_alerta", Float),
    Column("alerta", Boolean),
)

TABLAS = {tabla.name: tabla for tabla in metadata.sorted_tables}


def inicializar_db(engine):
    """Crea las tablas faltantes y agrega solo columnas nuevas conocidas.

    No hace DROP ni reemplaza tablas. Una columna nueva sin valor histórico
    queda NULL: no se inventan inscripciones a finales iguales a cero.
    """
    metadata.create_all(engine)
    nuevas = {
        "historico_anual": ["inscripciones_finales"],
        "predicciones": [
            "bandas_validadas", "umbral_referencia",
            "explicaciones_json", "valor_base_log_odds",
        ],
        "metricas_modelo": [
            columna.name for columna in metricas.columns
            if columna.name not in {
                "id_metrica", "fecha", "modelo_version",
                "roc_auc", "pr_auc", "recall", "precision", "umbral",
            }
        ],
    }

    with engine.begin() as conexion:
        for nombre_tabla, columnas in nuevas.items():
            existentes = {
                c["name"] for c in inspect(conexion).get_columns(nombre_tabla)
            }
            for nombre_columna in columnas:
                if nombre_columna in existentes:
                    continue
                tipo = TABLAS[nombre_tabla].c[nombre_columna].type.compile(
                    dialect=engine.dialect
                )
                # Tabla/columna provienen del esquema interno, nunca del usuario.
                conexion.execute(text(
                    f"ALTER TABLE {nombre_tabla} ADD COLUMN {nombre_columna} {tipo}"
                ))

    # En una base anterior pueden faltar estos índices aunque la tabla exista.
    # Si ya hay duplicados antiguos, la base rechazará el índice: no los borra.
    indice_predicciones.create(engine, checkfirst=True)
    indice_metricas.create(engine, checkfirst=True)
