"""Consultas y escrituras idempotentes para PostgreSQL y la demo SQLite."""
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db.connection import engine as engine_predeterminado
from src.db.schema import TABLAS, inicializar_db


def leer(tabla: str, engine=None) -> pd.DataFrame:
    """Solo admite las tablas conocidas; el nombre no se interpola en SQL."""
    if tabla not in TABLAS:
        raise ValueError(f"Tabla no permitida: {tabla}")
    return pd.read_sql(select(TABLAS[tabla]), engine or engine_predeterminado)


def upsert_df(df: pd.DataFrame, tabla: str, claves: list[str], conexion):
    """Inserta o actualiza registros por su clave, dentro de una transacción."""
    if df.empty:
        return
    destino = TABLAS[tabla]
    if set(df.columns) - set(destino.c.keys()):
        raise ValueError(f"Columnas desconocidas para {tabla}.")
    if not set(claves).issubset(df.columns):
        raise ValueError(f"Faltan claves de escritura para {tabla}.")
    if df[claves].isna().any().any() or df.duplicated(claves).any():
        raise ValueError(f"Claves nulas o duplicadas en {tabla}.")

    dialecto = conexion.dialect.name
    constructor = {"postgresql": pg_insert, "sqlite": sqlite_insert}.get(dialecto)
    if constructor is None:
        raise ValueError("Esta escritura admite PostgreSQL o SQLite.")

    # None se convierte en SQL NULL; evita enviar NaN como edad desconocida.
    registros = df.astype(object).where(pd.notna(df), None).to_dict("records")
    for inicio in range(0, len(registros), 200):
        sentencia = constructor(destino).values(registros[inicio:inicio + 200])
        cambios = {
            columna: sentencia.excluded[columna]
            for columna in df.columns if columna not in claves
        }
        if cambios:
            sentencia = sentencia.on_conflict_do_update(
                index_elements=claves, set_=cambios
            )
        else:
            sentencia = sentencia.on_conflict_do_nothing(index_elements=claves)
        conexion.execute(sentencia)


def cargar_historial(estudiantes: pd.DataFrame, historico: pd.DataFrame, engine=None):
    """Carga padres e historia juntos sin perder las claves foráneas."""
    db = engine or engine_predeterminado
    inicializar_db(db)
    with db.begin() as conexion:
        upsert_df(estudiantes, "estudiantes", ["id_alumno"], conexion)
        upsert_df(
            historico, "historico_anual",
            ["id_alumno", "anio_academico"], conexion,
        )


def guardar_predicciones(df: pd.DataFrame, engine=None):
    db = engine or engine_predeterminado
    inicializar_db(db)
    with db.begin() as conexion:
        upsert_df(
            df, "predicciones",
            ["id_alumno", "anio_base", "modelo_version", "fecha_scoring"],
            conexion,
        )


def guardar_metricas(df: pd.DataFrame, engine=None):
    db = engine or engine_predeterminado
    inicializar_db(db)
    with db.begin() as conexion:
        upsert_df(
            df, "metricas_modelo",
            ["modelo_version", "anio_evaluacion"], conexion,
        )


def insertar_df(df: pd.DataFrame, tabla: str, if_exists: str = "append", engine=None):
    """Compatibilidad para registrar drift; no permite borrar tablas."""
    if tabla not in TABLAS or if_exists != "append":
        raise ValueError("Solo se permite agregar datos a una tabla conocida.")
    db = engine or engine_predeterminado
    inicializar_db(db)
    with db.begin() as conexion:
        df.to_sql(tabla, conexion, if_exists="append", index=False)


def ultimas_predicciones(engine=None) -> pd.DataFrame:
    """Último año puntuado por estudiante, con desempate de fecha e identificador."""
    p, e = TABLAS["predicciones"], TABLAS["estudiantes"]
    ordenadas = select(
        p,
        func.row_number().over(
            partition_by=p.c.id_alumno,
            order_by=[
                p.c.anio_base.desc(), p.c.fecha_scoring.desc(),
                p.c.id_prediccion.desc(),
            ],
        ).label("orden"),
    ).subquery()
    consulta = (
        select(*[ordenadas.c[c.name] for c in p.columns], e.c.cohorte_carrera)
        .join(e, ordenadas.c.id_alumno == e.c.id_alumno)
        .where(ordenadas.c.orden == 1)
        .order_by(ordenadas.c.id_alumno)
    )
    return pd.read_sql(consulta, engine or engine_predeterminado)
