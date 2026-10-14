"""
Métricas de DATA DRIFT (PSI e classificação de severidade).

Puro numpy — sem dependência de spark ou dbutils — para rodar em pytest
sem precisar de um cluster Databricks. Usado por
notebooks/monitoracao/01_drift_dados.py.

CORREÇÃO: features de contagem concentradas em zero (ex.: atrasos de 60-89
e 90+ dias, ~95% de zeros) faziam os decis da referência colapsarem em um
único valor, e a versão anterior devolvia PSI = 0 por construção — um drift
real nessas variáveis passava despercebido. Agora, quando os decis têm
empates e a feature é discreta, os bins passam a ser os próprios valores
distintos (0, 1, 2, ..., com a cauda agrupada no último bin).
"""

import numpy as np

# Acima disso, a feature é tratada como contínua mesmo com empates nos decis
MAX_VALORES_DISCRETOS = 50


def _cortes_discretos(referencia: np.ndarray, n_bins: int) -> np.ndarray:
    """Um bin por valor distinto da referência (até n_bins); o último bin
    agrupa a cauda. Cortes nos pontos médios entre valores consecutivos."""
    valores = np.unique(referencia)
    n_cortes = min(len(valores) - 1, n_bins - 1)
    medios = (valores[:n_cortes] + valores[1 : n_cortes + 1]) / 2
    return np.concatenate([[-np.inf], medios, [np.inf]])


def _cortes(referencia: np.ndarray, n_bins: int) -> np.ndarray | None:
    """Decis da referência; valores distintos se os decis colapsarem numa
    feature discreta. None se a feature for constante (PSI não informativo)."""
    if len(np.unique(referencia)) < 2:
        return None

    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    decis_com_empate = len(cortes) < n_bins + 1
    if decis_com_empate and len(np.unique(referencia)) <= MAX_VALORES_DISCRETOS:
        return _cortes_discretos(referencia, n_bins)

    if len(cortes) < 3:
        # Contínua, mas com massa quase toda num ponto: separa esse ponto do resto
        return _cortes_discretos(referencia, 2)
    cortes[0], cortes[-1] = -np.inf, np.inf
    return cortes


def calcular_psi(referencia: np.ndarray, atual: np.ndarray, n_bins: int = 10) -> float:
    """PSI clássico: bins definidos pela referência, comparando a proporção
    de cada bin entre referência e atual. PSI = 0 quando as duas
    distribuições são idênticas nos mesmos cortes; cresce conforme elas
    se afastam.
    """
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)
    referencia = referencia[~np.isnan(referencia)]
    atual = atual[~np.isnan(atual)]
    if len(referencia) == 0 or len(atual) == 0:
        return float("nan")

    cortes = _cortes(referencia, n_bins)
    if cortes is None:
        return 0.0

    freq_ref, _ = np.histogram(referencia, bins=cortes)
    freq_atual, _ = np.histogram(atual, bins=cortes)

    prop_ref = np.clip(freq_ref / freq_ref.sum(), 1e-4, None)
    prop_atual = np.clip(freq_atual / freq_atual.sum(), 1e-4, None)

    return float(np.sum((prop_atual - prop_ref) * np.log(prop_atual / prop_ref)))


def classificar_psi(psi: float, moderado: float = 0.10, severo: float = 0.25) -> str:
    """Classifica um valor de PSI já calculado em estavel / moderado / SEVERO."""
    if np.isnan(psi):
        return "indefinido"
    if psi >= severo:
        return "SEVERO"
    if psi >= moderado:
        return "moderado"
    return "estavel"
