"""
Lógica de CONCEPT DRIFT: avaliação da queda de AUC-ROC vs. referência, e o
gerador sintético usado em SIMULATION_MODE.

Puro numpy/pandas — sem spark/dbutils/mlflow — para rodar em pytest sem
cluster. Usado por notebooks/monitoracao/02_concept_drift.py.

POR QUE O GERADOR FOI REFEITO
A primeira versão enfraquecia debt_ratio e revolving_utilization, mas:
  - debt_ratio é uma variável fraca no modelo real (5ª no ranking SHAP);
  - a utilização sintética vinha de beta(2, 5), quase toda abaixo de 40%,
    faixa em que o SHAP de revolving_utilization é praticamente plano;
  - os atrasos (2º sinal mais forte do modelo) não mudavam no drift.
Resultado: o drift mexia justamente onde o modelo não olha, e a AUC caía
só 0,0148 — abaixo do limiar de 0,03, sem alerta.

Agora o drift atua nos sinais COMPORTAMENTAIS que o modelo de fato usa
(utilização do limite e histórico de atrasos), com intensidade graduável,
e a distribuição de utilização inclui clientes perto e acima do limite,
como na base real.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "age",
    "num_dependents",
    "monthly_income",
    "debt_ratio",
    "revolving_utilization",
    "num_open_credit_lines",
    "num_real_estate_loans",
    "num_times_30_59_days_late",
    "num_times_60_89_days_late",
    "num_times_90_days_late",
    "total_delinquency_events",
]

# Taxa de inadimplência mantida constante em todos os lotes (≈ base real,
# 6,68%). Assim a simulação isola a mudança em P(alvo | features) — concept
# drift puro — sem misturar uma mudança de prevalência.
TAXA_ALVO = 0.0668


def avaliar_queda_auc(auc_referencia: float, auc_atual: float, limiar: float = 0.03) -> dict:
    """Compara a AUC-ROC atual contra a referência e decide se dispara alerta
    de concept drift. Retorna a queda (positiva = piorou) e o booleano de alerta.
    """
    queda = float(auc_referencia - auc_atual)
    return {"queda_auc": queda, "alerta": queda >= limiar}


def _gerar_features(r: np.random.Generator, n: int) -> dict:
    """Distribuições inspiradas na base real. Não dependem da intensidade de
    drift: com a mesma seed, as features são idênticas em qualquer cenário."""
    # Mistura: maioria com uso baixo, uma parte com uso médio e uma parte
    # perto/acima do limite — a faixa em que o modelo real mais reage (H1).
    grupo = r.choice(3, size=n, p=[0.55, 0.30, 0.15])
    revolving = np.where(
        grupo == 0,
        r.beta(0.8, 4.0, n),
        np.where(grupo == 1, r.beta(2.0, 2.0, n), r.uniform(0.9, 1.1, n)),
    )

    # Atrasos concentrados em zero, como na base real
    n30 = r.poisson(0.25, n).clip(0, 10)
    n60 = r.poisson(0.08, n).clip(0, 10)
    n90 = r.poisson(0.09, n).clip(0, 10)

    return {
        "age": r.normal(52, 15, n).clip(21, 100),
        "num_dependents": r.poisson(0.75, n).clip(0, 8),
        "monthly_income": r.lognormal(8.6, 0.6, n),
        "debt_ratio": r.gamma(2.0, 0.2, n).clip(0, 5),
        "revolving_utilization": revolving,
        "num_open_credit_lines": r.poisson(8, n).clip(0, 30),
        "num_real_estate_loans": r.poisson(1.0, n).clip(0, 6),
        "num_times_30_59_days_late": n30,
        "num_times_60_89_days_late": n60,
        "num_times_90_days_late": n90,
        "total_delinquency_events": n30 + n60 + n90,
    }


def _intercepto_para_taxa(sinal: np.ndarray, taxa: float) -> float:
    """Encontra, por bisseção, o intercepto que faz a média de
    sigmoid(intercepto + sinal) igual à taxa desejada."""
    baixo, alto = -30.0, 30.0
    for _ in range(80):
        meio = (baixo + alto) / 2
        if np.mean(1 / (1 + np.exp(-(meio + sinal)))) > taxa:
            alto = meio
        else:
            baixo = meio
    return (baixo + alto) / 2


def gerar_lote_sintetico(
    n: int,
    concept_drift: bool = False,
    seed: int = 0,
    target_col: str = "target_dlq_2yrs",
    intensidade: float | None = None,
) -> pd.DataFrame:
    """Gera um lote sintético de clientes de crédito.

    intensidade: 0.0 = relação feature->alvo igual à de referência;
                 1.0 = os sinais comportamentais (utilização do limite e
                 atrasos) deixam de explicar o alvo, que passa a depender
                 de perfil (dependentes, imóveis, dívida/renda).
                 Valores intermediários interpolam linearmente.
    concept_drift: atalho mantido por compatibilidade — equivale a
                 intensidade=1.0 (True) ou 0.0 (False). Ignorado quando
                 `intensidade` é informada.
    """
    if intensidade is None:
        intensidade = 1.0 if concept_drift else 0.0
    if not 0.0 <= intensidade <= 1.0:
        raise ValueError(f"intensidade deve estar entre 0 e 1, recebido {intensidade}")

    r = np.random.default_rng(seed)
    f = _gerar_features(r, n)

    # Sinais que o modelo real mais usa (ranking SHAP: utilização e atrasos)
    sinal_comportamental = (
        2.8 * f["revolving_utilization"]
        + 1.5 * (f["revolving_utilization"] >= 0.9)
        + 0.55 * f["num_times_30_59_days_late"]
        + 0.85 * f["num_times_60_89_days_late"]
        + 1.05 * f["num_times_90_days_late"]
    )
    # Sinais de perfil, fracos na referência
    sinal_perfil = -0.02 * (f["age"] - 52) + 0.3 * f["debt_ratio"]
    # Sinal que só passa a existir com o drift
    sinal_novo = 0.6 * f["num_dependents"] + 0.5 * f["num_real_estate_loans"]

    sinal = (1 - intensidade) * sinal_comportamental + sinal_perfil + intensidade * sinal_novo
    logit = _intercepto_para_taxa(sinal, TAXA_ALVO) + sinal

    target = r.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({**f, target_col: target})[FEATURE_COLS + [target_col]]
