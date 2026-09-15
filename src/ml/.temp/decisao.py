"""
Regra de decisão de aprovação de crédito a partir da probabilidade do modelo.

Puro Python — sem spark/dbutils/mlflow — para rodar em pytest sem cluster.
Usado por notebooks/ml/03_inferencia_batch.py.
"""


def classificar_decisao(probabilidade: float, threshold: float) -> str:
    """'negado' quando a probabilidade de inadimplência é >= threshold,
    'aprovado' caso contrário. Mesma regra usada no Q2 do README (threshold
    ótimo por custo assimétrico)."""
    if not 0.0 <= probabilidade <= 1.0:
        raise ValueError(f"probabilidade fora do intervalo [0, 1]: {probabilidade}")
    return "negado" if probabilidade >= threshold else "aprovado"
