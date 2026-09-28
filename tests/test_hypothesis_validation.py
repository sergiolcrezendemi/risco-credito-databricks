"""
Testes de src/models/hypothesis_validation.py — rodam sem cluster.
"""

import numpy as np
import pandas as pd
import shap

from src.models import hypothesis_validation as hv


def _shap_monotonico_com_outliers(n=2_000, seed=0):
    """Utilização com relação crescente e saturada com o SHAP, mais 1% de
    outliers extremos (como na base real, com valores na casa dos milhares).
    Nos outliers, que são erros de cadastro, o SHAP não segue a tendência."""
    rng = np.random.default_rng(seed)
    utilizacao = rng.uniform(0, 1.2, n)
    shap_util = np.tanh(3 * (utilizacao - 0.4)) + rng.normal(0, 0.05, n)
    outliers = rng.choice(n, size=n // 100, replace=False)
    utilizacao[outliers] = rng.uniform(500, 5_000, outliers.size)
    shap_util[outliers] = rng.normal(0, 0.5, outliers.size)

    sample = pd.DataFrame({"revolving_utilization": utilizacao, "age": rng.uniform(21, 90, n)})
    valores = np.column_stack([shap_util, rng.normal(0, 0.1, n)])
    explicacao = shap.Explanation(values=valores, data=sample.values, feature_names=list(sample))
    ranking = pd.DataFrame(
        {"feature": ["revolving_utilization", "age"], "mean_abs_shap": np.abs(valores).mean(axis=0)}
    )
    return sample, explicacao, ranking


def test_h1_detecta_relacao_monotonica_mesmo_com_outliers():
    """Com Pearson, os outliers achatavam a correlação e a H1 saía
    'Contrariada' mesmo com a relação claramente crescente."""
    sample, explicacao, ranking = _shap_monotonico_com_outliers()

    pearson = np.corrcoef(sample["revolving_utilization"], explicacao[:, 0].values)[0, 1]
    resultado = hv.validate_h1(sample, explicacao, ranking)

    assert pearson < 0.05  # o problema que existia
    assert resultado["correlacao"] > 0.5
    assert resultado["status"] == "Confirmada"
    assert resultado["metodo"] == "Spearman"


def test_h1_nao_confirma_quando_nao_ha_relacao():
    rng = np.random.default_rng(1)
    n = 2_000
    sample = pd.DataFrame({"revolving_utilization": rng.uniform(0, 1, n)})
    explicacao = shap.Explanation(
        values=rng.normal(0, 1, (n, 1)), data=sample.values, feature_names=list(sample)
    )
    ranking = pd.DataFrame({"feature": ["revolving_utilization"], "mean_abs_shap": [0.8]})

    assert hv.validate_h1(sample, explicacao, ranking)["status"] != "Confirmada"
