# Memoria del proyecto (borrador)

## 1. Problema
Tasa de abandono en Lic. en Psicología; alerta temprana a 2 años.

## 2. Datos
1.523 estudiantes, 3.965 registros estudiante-año (2000-2026). Anonimizados.

## 3. Metodología
- Variable objetivo: sin actividad en t+1 y t+2. Censurado excluido.
- Split por estudiante (GroupKFold). Features anti-leakage documentadas.
- Baseline logístico vs LightGBM; calibración; SHAP.

## 4. Producto
API en Railway, modelo versionado en Hugging Face Hub, dashboard Power BI.

## 5. Limitaciones
Tamaño del dataset, censurado, posibles sesgos de registro.

## 6. Decisiones de despliegue (análisis de alternativas)
Railway (elegido, estable, ya contratado) vs Render free (se duerme) vs HF Spaces.
DB: Railway PostgreSQL vs Supabase/Neon free. Modelo: HF Hub (gratis, versionado).
