"""
Testes do parâmetro `intensidade` de gerar_lote_sintetico.
Complementam tests/test_concept_drift.py — rodam sem cluster (pytest puro).
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.monitoring.concept_drift import FEATURE_COLS, TAXA_ALVO, gerar_lote_sintetico

TARGET = "target_dlq_2yrs"


def test_mesma_seed_gera_mesmas_features_em_qualquer_intensidade():
    """Concept drift puro: só P(alvo | features) muda; as features não."""
    sem_drift = gerar_lote_sintetico(2_000, intensidade=0.0, seed=7)
    com_drift = gerar_lote_sintetico(2_000, intensidade=1.0, seed=7)
    assert sem_drift[FEATURE_COLS].equals(com_drift[FEATURE_COLS])


@pytest.mark.parametrize("intensidade", [0.0, 0.5, 1.0])
def test_taxa_de_inadimplencia_se_mantem(intensidade):
    """A prevalência fica constante, para não confundir drift de conceito
    com mudança na taxa de inadimplência."""
    lote = gerar_lote_sintetico(50_000, intensidade=intensidade, seed=3)
    assert lote[TARGET].mean() == pytest.approx(TAXA_ALVO, abs=0.005)


def test_parametro_antigo_concept_drift_continua_funcionando():
    via_bool = gerar_lote_sintetico(500, True, 1)
    via_intensidade = gerar_lote_sintetico(500, seed=1, intensidade=1.0)
    assert via_bool.equals(via_intensidade)


@pytest.mark.parametrize("valor", [-0.1, 1.5])
def test_intensidade_fora_do_intervalo_falha(valor):
    with pytest.raises(ValueError):
        gerar_lote_sintetico(100, seed=0, intensidade=valor)


def test_auc_cai_conforme_intensidade_aumenta():
    """Um modelo treinado sem drift deve piorar monotonicamente com o drift."""
    treino = gerar_lote_sintetico(30_000, intensidade=0.0, seed=10)
    modelo = LogisticRegression(max_iter=2_000).fit(treino[FEATURE_COLS], treino[TARGET])

    aucs = []
    for i, k in enumerate([0.0, 0.5, 1.0]):
        lote = gerar_lote_sintetico(30_000, intensidade=k, seed=20 + i)
        aucs.append(roc_auc_score(lote[TARGET], modelo.predict_proba(lote[FEATURE_COLS])[:, 1]))

    assert aucs[0] > aucs[1] > aucs[2]
    assert aucs[0] - aucs[2] > 0.10
