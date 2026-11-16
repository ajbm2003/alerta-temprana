"""Metricas y calibracion; resultados se persisten en tabla metricas_modelo."""
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score


def evaluar(y_true, y_prob, umbral: float = 0.5) -> dict:
    y_pred = (y_prob >= umbral).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "umbral": umbral,
    }
