"""
Testes para src/risco_credito/monitoring/data_drift.py
"""
import numpy as np
import pytest

from risco_credito.monitoring.data_drift import calcular_psi, classificar_psi


def test_psi_distribuicoes_identicas_e_proximo_de_zero():
    rng = np.random.default_rng(42)
    referencia = rng.normal(0, 1, 5000)
    atual = referencia.copy()
    psi = calcular_psi(referencia, atual)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_cresce_com_o_deslocamento_da_distribuicao():
    rng = np.random.default_rng(42)
    referencia = rng.normal(0, 1, 5000)
    atual_leve = rng.normal(0.3, 1, 5000)
    atual_forte = rng.normal(2.0, 1, 5000)

    psi_leve = calcular_psi(referencia, atual_leve)
    psi_forte = calcular_psi(referencia, atual_forte)

    assert psi_leve > 0
    assert psi_forte > psi_leve, "deslocamento maior deve produzir PSI maior"


def test_psi_ignora_nans():
    rng = np.random.default_rng(0)
    referencia = rng.normal(0, 1, 2000)
    atual = np.concatenate([rng.normal(0, 1, 2000), [np.nan, np.nan]])
    psi = calcular_psi(referencia, atual)
    assert not np.isnan(psi)


def test_psi_referencia_quase_constante_nao_quebra():
    referencia = np.full(1000, 5.0)
    atual = np.full(1000, 5.0) + np.random.default_rng(1).normal(0, 0.01, 1000)
    psi = calcular_psi(referencia, atual)
    assert psi == 0.0  # menos de 3 cortes distintos -> não informativo, retorna 0


def test_psi_arrays_vazios_retorna_nan():
    assert np.isnan(calcular_psi(np.array([]), np.array([1.0, 2.0])))
    assert np.isnan(calcular_psi(np.array([1.0, 2.0]), np.array([])))


@pytest.mark.parametrize(
    "psi, esperado",
    [
        (0.0, "estavel"),
        (0.05, "estavel"),
        (0.099, "estavel"),
        (0.10, "moderado"),
        (0.20, "moderado"),
        (0.249, "moderado"),
        (0.25, "SEVERO"),
        (1.5, "SEVERO"),
    ],
)
def test_classificar_psi_limiares(psi, esperado):
    assert classificar_psi(psi) == esperado


def test_classificar_psi_nan_e_indefinido():
    assert classificar_psi(float("nan")) == "indefinido"


def test_classificar_psi_limiares_customizados():
    assert classificar_psi(0.05, moderado=0.03, severo=0.08) == "moderado"
    assert classificar_psi(0.09, moderado=0.03, severo=0.08) == "SEVERO"
