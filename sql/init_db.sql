-- Esquema PostgreSQL equivalente a src/db/schema.py.
-- La carga Python crea/migra estas tablas automáticamente sin hacer DROP.


CREATE TABLE IF NOT EXISTS drift (
	id_drift SERIAL NOT NULL, 
	fecha DATE, 
	variable VARCHAR(100), 
	metrica_drift FLOAT, 
	umbral_alerta FLOAT, 
	alerta BOOLEAN, 
	PRIMARY KEY (id_drift)
);

CREATE TABLE IF NOT EXISTS estudiantes (
	id_alumno SERIAL NOT NULL, 
	cohorte_carrera INTEGER, 
	cohorte_universidad INTEGER, 
	tuvo_actividad_previa INTEGER, 
	anios_actividad_previa INTEGER, 
	PRIMARY KEY (id_alumno)
);

CREATE TABLE IF NOT EXISTS metricas_modelo (
	id_metrica SERIAL NOT NULL, 
	fecha DATE, 
	modelo_version VARCHAR(80), 
	anio_evaluacion INTEGER, 
	umbral FLOAT, 
	evaluacion_tipo VARCHAR(80), 
	bootstrap_muestras INTEGER, 
	roc_auc FLOAT, 
	roc_auc_ic95_inferior FLOAT, 
	roc_auc_ic95_superior FLOAT, 
	pr_auc FLOAT, 
	pr_auc_ic95_inferior FLOAT, 
	pr_auc_ic95_superior FLOAT, 
	recall FLOAT, 
	recall_ic95_inferior FLOAT, 
	recall_ic95_superior FLOAT, 
	precision FLOAT, 
	precision_ic95_inferior FLOAT, 
	precision_ic95_superior FLOAT, 
	brier FLOAT, 
	brier_ic95_inferior FLOAT, 
	brier_ic95_superior FLOAT, 
	log_loss FLOAT, 
	log_loss_ic95_inferior FLOAT, 
	log_loss_ic95_superior FLOAT, 
	PRIMARY KEY (id_metrica)
);

CREATE TABLE IF NOT EXISTS historico_anual (
	id_alumno INTEGER NOT NULL, 
	anio_academico INTEGER NOT NULL, 
	edad FLOAT, 
	asignaturas_cursadas INTEGER, 
	asignaturas_aprobadas INTEGER, 
	inscripciones_finales INTEGER, 
	tasa_aprobacion_asignaturas FLOAT, 
	tasa_desaprobacion_asignaturas FLOAT, 
	finales_aprobados INTEGER, 
	finales_ausentes INTEGER, 
	tasa_ausencias_finales FLOAT, 
	equivalencias_asignaturas INTEGER, 
	porcentaje_avance_cursado FLOAT, 
	porcentaje_avance_carrera FLOAT, 
	cantidad_pausas INTEGER, 
	maxima_duracion_pausa INTEGER, 
	antiguedad_carrera INTEGER, 
	cambio_plan INTEGER, 
	PRIMARY KEY (id_alumno, anio_academico), 
	FOREIGN KEY(id_alumno) REFERENCES estudiantes (id_alumno)
);

CREATE TABLE IF NOT EXISTS predicciones (
	id_prediccion SERIAL NOT NULL, 
	id_alumno INTEGER NOT NULL, 
	anio_base INTEGER NOT NULL, 
	probabilidad_abandono FLOAT, 
	banda_riesgo VARCHAR(10), 
	factor_1 TEXT, 
	factor_2 TEXT, 
	factor_3 TEXT, 
	modelo_version VARCHAR(80) NOT NULL, 
	fecha_scoring DATE NOT NULL, 
	bandas_validadas BOOLEAN, 
	umbral_referencia FLOAT, 
	explicaciones_json TEXT, 
	valor_base_log_odds FLOAT, 
	PRIMARY KEY (id_prediccion), 
	FOREIGN KEY(id_alumno) REFERENCES estudiantes (id_alumno)
);

-- Compatibilidad con bases creadas antes de la logística con recuentos.

ALTER TABLE historico_anual ADD COLUMN IF NOT EXISTS inscripciones_finales INTEGER;

ALTER TABLE predicciones ADD COLUMN IF NOT EXISTS bandas_validadas BOOLEAN;

ALTER TABLE predicciones ADD COLUMN IF NOT EXISTS umbral_referencia FLOAT;

ALTER TABLE predicciones ADD COLUMN IF NOT EXISTS explicaciones_json TEXT;

ALTER TABLE predicciones ADD COLUMN IF NOT EXISTS valor_base_log_odds FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS anio_evaluacion INTEGER;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS evaluacion_tipo VARCHAR(80);

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS bootstrap_muestras INTEGER;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS roc_auc_ic95_inferior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS roc_auc_ic95_superior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS pr_auc_ic95_inferior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS pr_auc_ic95_superior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS recall_ic95_inferior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS recall_ic95_superior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS precision_ic95_inferior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS precision_ic95_superior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS brier FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS brier_ic95_inferior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS brier_ic95_superior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS log_loss FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS log_loss_ic95_inferior FLOAT;

ALTER TABLE metricas_modelo ADD COLUMN IF NOT EXISTS log_loss_ic95_superior FLOAT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_metricas_modelo_evaluacion ON metricas_modelo (modelo_version, anio_evaluacion);

CREATE UNIQUE INDEX IF NOT EXISTS uq_predicciones_scoring ON predicciones (id_alumno, anio_base, modelo_version, fecha_scoring);
