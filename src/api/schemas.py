"""Contrato explícito: no se inventan valores cero cuando falta información."""
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class HistoriaAnual(BaseModel):
    # Se conserva "anio" como alias para los clientes anteriores.
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)

    anio_academico: int = Field(
        validation_alias=AliasChoices("anio_academico", "anio"), ge=1900, le=2100
    )
    edad: float | None = Field(default=None, ge=0, le=130)
    antiguedad_carrera: int = Field(ge=0)
    asignaturas_cursadas: int = Field(ge=0)
    asignaturas_aprobadas: int | None = Field(default=None, ge=0)
    inscripciones_finales: int = Field(ge=0)
    finales_aprobados: int = Field(ge=0)
    tasa_aprobacion_asignaturas: float = Field(ge=0, le=1)
    tasa_desaprobacion_asignaturas: float = Field(ge=0, le=1)
    tasa_ausencias_finales: float = Field(ge=0, le=1)
    porcentaje_avance_carrera: float = Field(ge=0, le=1)
    porcentaje_avance_cursado: float = Field(ge=0, le=1)
    cantidad_pausas: int = Field(ge=0)
    maxima_duracion_pausa: int = Field(ge=0)

    @model_validator(mode="after")
    def verificar_recuentos(self):
        if (
            self.asignaturas_aprobadas is not None
            and self.asignaturas_aprobadas > self.asignaturas_cursadas
        ):
            raise ValueError("Las asignaturas aprobadas no pueden superar las cursadas.")
        if self.finales_aprobados > self.inscripciones_finales:
            raise ValueError("Los finales aprobados no pueden superar las inscripciones.")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_alumno: int = Field(gt=0, description="Identificador anonimizado")
    anio_corte: int | None = Field(
        default=None, ge=1900, le=2100,
        description="Año completo a predecir. Por defecto, el cierre configurado.",
    )
    historia: list[HistoriaAnual] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def verificar_anios_unicos(self):
        anios = [registro.anio_academico for registro in self.historia]
        if len(anios) != len(set(anios)):
            raise ValueError("No puede repetirse un año dentro de la historia.")
        return self


class FactorExplicativo(BaseModel):
    feature: str
    valor_usado: float
    contribucion_log_odds: float
    efecto: Literal["aumenta", "reduce", "neutro"]


class PredictResponse(BaseModel):
    id_alumno: int
    anio_base: int
    probabilidad_abandono: float = Field(ge=0, le=1)
    banda_riesgo: Literal["bajo", "medio", "alto"]
    modelo_version: str
    bandas_validadas: bool
    umbral_referencia: float
    factores_principales: list[str]
    factores_detalle: list[FactorExplicativo]
    valor_base_log_odds: float


class BatchResponse(BaseModel):
    registros_scorados: int
    fecha: str
