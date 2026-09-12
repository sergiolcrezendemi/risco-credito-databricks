"""
Testes para src/risco_credito/monitoring/concept_drift.py
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from risco_credito.monitoring.concept_drift import (
    FEATURE_COLS,
    avaliar_queda_auc,
    gerar_lote_sintetico,
)


def test_avaliar_queda_auc_sem_alerta_quando_auc_estavel():
    resultado = avaliar_queda_auc(auc_referencia=0.80, auc_atual=0.795, limiar=0.03)
    assert resultado["queda_auc"] == pytest.approx(0.005, abs=1e-9)
    assert resultado["alerta"] is False


def test_avaliar_queda_auc_dispara_alerta_no_limiar_exato():
    resultado = avaliar_queda_auc(auc_referencia=0.80, auc_atual=0.77, limiar=0.03)
    assert resultado["queda_auc"] == pytest.approx(0.03, abs=1e-9)
    assert resultado["alerta"] is True  # >= limiar, não só >


def test_avaliar_queda_auc_auc_melhorou_fica_negativa_sem_alerta():
    resultado = avaliar_queda_auc(auc_referencia=0.80, auc_atual=0.85, limiar=0.03)
    assert resultado["queda_auc"] < 0
    assert resultado["alerta"] is False


def test_gerar_lote_sintetico_colunas_e_tamanho():
    df = gerar_lote_sintetico(n=500, concept_drift=False, seed=1)
    assert len(df) == 500
    for col in FEATURE_COLS:
        assert col in df.columns
    assert "target_dlq_2yrs" in df.columns
    assert df["target_dlq_2yrs"].isin([0, 1]).all()


def test_gerar_lote_sintetico_reprodutivel_com_mesma_seed():
    df1 = gerar_lote_sintetico(n=200, concept_drift=False, seed=7)
    df2 = gerar_lote_sintetico(n=200, concept_drift=False, seed=7)
    pd_testing_equal = (df1.to_numpy() == df2.to_numpy()).all()
    assert pd_testing_equal, "mesma seed deve gerar o mesmo lote (essencial pro SIMULATION_MODE ser comparável)"


def test_gerar_lote_sintetico_sem_nans():
    df = gerar_lote_sintetico(n=300, concept_drift=True, seed=3)
    assert not df.isna().any().any()


def test_concept_drift_reduz_auc_de_um_modelo_treinado_sem_drift():
    """Teste de integração leve: treina um modelo simples na referência
    (sem drift) e confirma que ele performa pior no lote com concept drift —
    é exatamente o comportamento que 02_concept_drift.py espera detectar."""
    referencia = gerar_lote_sintetico(n=4000, concept_drift=False, seed=1)
    lote_drift = gerar_lote_sintetico(n=4000, concept_drift=True, seed=2)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(referencia[FEATURE_COLS], referencia["target_dlq_2yrs"])

    auc_referencia = roc_auc_score(
        referencia["target_dlq_2yrs"], modelo.predict_proba(referencia[FEATURE_COLS])[:, 1]
    )
    auc_drift = roc_auc_score(
        lote_drift["target_dlq_2yrs"], modelo.predict_proba(lote_drift[FEATURE_COLS])[:, 1]
    )

    resultado = avaliar_queda_auc(auc_referencia, auc_drift, limiar=0.03)
    assert resultado["queda_auc"] > 0, "modelo treinado sem drift deve performar pior no lote com drift"
