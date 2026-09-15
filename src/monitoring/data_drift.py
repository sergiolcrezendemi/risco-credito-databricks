"""
Métricas de DATA DRIFT (PSI e classificação de severidade).

Puro numpy — sem dependência de spark ou dbutils — para rodar em pytest
sem precisar de um cluster Databricks. Usado por
notebooks/monitoracao/01_drift_dados.py.
"""
import numpy as np


def calcular_psi(referencia: np.ndarray, atual: np.ndarray, n_bins: int = 10) -> float:
    """PSI clássico: bins definidos pelos decis da referência, comparando a
    proporção de cada bin entre referência e atual. PSI = 0 quando as duas
    distribuições são idênticas nos mesmos cortes; cresce conforme elas
    se afastam.
    """
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)
    referencia = referencia[~np.isnan(referencia)]
    atual = atual[~np.isnan(atual)]
    if len(referencia) == 0 or len(atual) == 0:
        return float("nan")

    quantis = np.linspace(0, 1, n_bins + 1)
    cortes = np.unique(np.quantile(referencia, quantis))
    if len(cortes) < 3:
        # Feature quase constante na referência (poucos valores distintos) — PSI não é informativo
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf

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
