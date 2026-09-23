"""
Testes de src/models/bias_roi_validation.py — rodam sem cluster.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import bias_roi_validation as brv


def test_renda_nao_informada_vira_faixa_propria():
    """Antes, renda nula era preenchida com 0 e caía na faixa de menor renda."""
    renda = pd.Series([np.nan, 1000, 2000, 3000, 4000, 5000, np.nan, 6000, 7000, 8000, 9000, 10000])
    faixas = pd.Series(brv.faixa_renda(renda))
    assert (faixas[renda.isna()] == "não informada").all()
    assert not (faixas[renda.notna()] == "não informada").any()
    assert faixas[1] == "Q1 (menor)"


def test_idade_nas_bordas_e_nula():
    faixas = brv.faixa_idade(pd.Series([18, 30, 31, 60, 61, 110, np.nan]))
    assert list(faixas) == ["18-30", "18-30", "31-40", "51-60", "60+", "60+", "não informada"]


def test_intervalo_wilson_contem_proporcao_e_fica_entre_0_e_1():
    baixo, alto = brv._intervalo_wilson(5, 100)
    assert 0 <= baixo < 0.05 < alto <= 1
    assert brv._intervalo_wilson(0, 50)[0] == pytest.approx(0.0, abs=1e-12)


def test_ponto_de_equilibrio_do_impacto_financeiro():
    """Com 2 maus recusados e 6 bons negados, o modelo se paga se a perda
    por calote for maior que 3x a margem."""
    df_val = pd.DataFrame(
        {"y_true": [1, 1, 0, 0, 0, 0, 0, 0, 0, 1], "y_pred": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]}
    )
    r = brv.financial_impact(df_val, custo_aprovar_mau=300, custo_negar_bom=100)
    assert r["razao_equilibrio_perda_margem"] == pytest.approx(3.0)
    assert r["impacto_liquido"] == pytest.approx(0.0)


def test_alerta_de_bons_negados_so_quando_o_grupo_e_de_fato_prejudicado():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"age": rng.choice([25, 45, 65], n), "monthly_income": rng.uniform(1e3, 1e4, n)})
    y = pd.Series(rng.binomial(1, 0.07, n))
    proba = rng.uniform(0, 0.4, n)
    # Bons pagadores de 18-30 recebem score alto: o modelo os recusa muito mais
    proba[(X["age"] == 25) & (y == 0)] += 0.3
    resultado = brv.bias_analysis(X, y, proba, threshold=0.45)
    tabela = resultado["tabela_idade"].set_index("grupo")
    assert tabela.loc["18-30", "bons_negados_acima_da_media"]
    assert not tabela.loc["41-50", "bons_negados_acima_da_media"]


def test_calibracao_isotonica_aproxima_media_prevista_da_observada():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.binomial(1, 0.07, 20_000))
    proba_inflada = np.clip(y * 0.3 + rng.uniform(0.1, 0.5, len(y)), 0, 1)  # superestima o risco
    r = brv.calibration_analysis(y, proba_inflada)
    assert r["bruta"]["media_prevista"] > 3 * r["inadimplencia_observada"]
    assert r["calibrada"]["media_prevista"] == pytest.approx(r["inadimplencia_observada"], abs=0.01)
    assert r["calibrada"]["brier"] < r["bruta"]["brier"]
