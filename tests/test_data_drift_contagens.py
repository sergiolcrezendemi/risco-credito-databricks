"""
Testes do PSI em features de contagem concentradas em zero — o caso que
fazia a versão anterior devolver PSI = 0 por construção.
Complementam tests/test_data_drift.py.
"""

import numpy as np
import pytest

from src.monitoring.data_drift import calcular_psi, classificar_psi


def _atrasos(rng, n, prop_com_atraso):
    """Contagem de atrasos: maioria zero, cauda de 1 a 4 (como 60-89 dias)."""
    tem_atraso = rng.random(n) < prop_com_atraso
    return np.where(tem_atraso, rng.integers(1, 5, n), 0)


def test_detecta_drift_em_feature_concentrada_em_zero():
    """Clientes com atraso passam de 5,5% para 20%. A versão anterior
    devolvia PSI = 0 ('estável'); agora o drift aparece."""
    rng = np.random.default_rng(0)
    referencia = _atrasos(rng, 100_000, 0.055)
    atual = _atrasos(rng, 100_000, 0.20)

    psi = calcular_psi(referencia, atual)
    assert psi > 0.10
    assert classificar_psi(psi) != "estavel"


def test_drift_maior_gera_alerta_severo():
    rng = np.random.default_rng(3)
    psi = calcular_psi(_atrasos(rng, 100_000, 0.055), _atrasos(rng, 100_000, 0.35))
    assert classificar_psi(psi) == "SEVERO"


def test_sem_falso_alarme_em_feature_concentrada_em_zero():
    rng = np.random.default_rng(1)
    psi = calcular_psi(_atrasos(rng, 100_000, 0.055), _atrasos(rng, 100_000, 0.055))
    assert classificar_psi(psi) == "estavel"


def test_feature_continua_mantem_o_comportamento():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 50_000)
    assert calcular_psi(ref, rng.normal(0, 1, 50_000)) < 0.01
    assert classificar_psi(calcular_psi(ref, rng.normal(1, 1, 50_000))) == "SEVERO"


def test_feature_constante_nao_e_informativa():
    assert calcular_psi(np.zeros(1_000), np.ones(1_000)) == pytest.approx(0.0)
