"""
Lógica de CONCEPT DRIFT: avaliação da queda de AUC-ROC vs. referência, e o
gerador sintético usado em SIMULATION_MODE.

Puro numpy/pandas — sem spark/dbutils/mlflow — para rodar em pytest sem
cluster. Usado por notebooks/monitoracao/02_concept_drift.py.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "age", "num_dependents", "monthly_income", "debt_ratio", "revolving_utilization",
    "num_open_credit_lines", "num_real_estate_loans", "num_times_30_59_days_late",
    "num_times_60_89_days_late", "num_times_90_days_late", "total_delinquency_events",
]


def avaliar_queda_auc(auc_referencia: float, auc_atual: float, limiar: float = 0.03) -> dict:
    """Compara a AUC-ROC atual contra a referência e decide se dispara alerta
    de concept drift. Retorna a queda (positiva = piorou) e o booleano de alerta.
    """
    queda = float(auc_referencia - auc_atual)
    return {"queda_auc": queda, "alerta": queda >= limiar}


def gerar_lote_sintetico(n: int, concept_drift: bool, seed: int, target_col: str = "target_dlq_2yrs") -> pd.DataFrame:
    """Gera um lote sintético de clientes de crédito. Com concept_drift=True,
    enfraquece o peso de debt_ratio/revolving_utilization na relação com o
    target — simula o cenário em que essas features (as mais preditivas do
    modelo real, ver README Q1) deixaram de prever tão bem quanto antes.
    """
    r = np.random.default_rng(seed)
    age = r.normal(45, 12, n).clip(21, 90)
    monthly_income = r.lognormal(8.6, 0.6, n)
    debt_ratio = r.gamma(2.0, 0.25, n).clip(0, 5)
    revolving_utilization = r.beta(2, 5, n).clip(0, 1.5)
    num_times_30_59 = r.poisson(0.35, n).clip(0, 10)
    num_times_60_89 = r.poisson(0.12, n).clip(0, 10)
    num_times_90 = r.poisson(0.08, n).clip(0, 10)
    num_open_credit_lines = r.poisson(8, n).clip(0, 30)

    coef_debt = 2.2 if not concept_drift else 0.7
    coef_revolv = 2.6 if not concept_drift else 0.8
    logit = (
        -3.6 + coef_debt * debt_ratio + coef_revolv * revolving_utilization
        + 0.55 * num_times_30_59 + 0.85 * num_times_60_89 + 1.05 * num_times_90
        - 0.00002 * monthly_income - 0.01 * age
    )
    target = r.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "age": age, "num_dependents": r.poisson(0.9, n).clip(0, 8), "monthly_income": monthly_income,
        "debt_ratio": debt_ratio, "revolving_utilization": revolving_utilization,
        "num_open_credit_lines": num_open_credit_lines, "num_real_estate_loans": r.poisson(1.0, n).clip(0, 6),
        "num_times_30_59_days_late": num_times_30_59, "num_times_60_89_days_late": num_times_60_89,
        "num_times_90_days_late": num_times_90,
        "total_delinquency_events": num_times_30_59 + num_times_60_89 + num_times_90,
        target_col: target,
    })
