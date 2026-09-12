"""
Testes para src/risco_credito/ml/decisao.py
"""
import pytest

from risco_credito.ml.decisao import classificar_decisao


def test_classificar_decisao_abaixo_do_threshold_e_aprovado():
    assert classificar_decisao(0.30, threshold=0.56) == "aprovado"


def test_classificar_decisao_acima_do_threshold_e_negado():
    assert classificar_decisao(0.80, threshold=0.56) == "negado"


def test_classificar_decisao_no_threshold_exato_e_negado():
    # >= threshold nega — mesma convenção usada no notebook (Q2 do README)
    assert classificar_decisao(0.56, threshold=0.56) == "negado"


@pytest.mark.parametrize("probabilidade", [0.0, 0.001, 0.999, 1.0])
def test_classificar_decisao_limites_do_intervalo(probabilidade):
    # não deve levantar exceção para os extremos válidos
    resultado = classificar_decisao(probabilidade, threshold=0.5)
    assert resultado in {"aprovado", "negado"}


@pytest.mark.parametrize("probabilidade_invalida", [-0.01, 1.01, 2.0, -5.0])
def test_classificar_decisao_rejeita_probabilidade_fora_do_intervalo(probabilidade_invalida):
    with pytest.raises(ValueError):
        classificar_decisao(probabilidade_invalida, threshold=0.5)
