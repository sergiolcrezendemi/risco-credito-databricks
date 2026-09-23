# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# =============================================================================
# notebooks/monitoracao/02_concept_drift.py
# -----------------------------------------------------------------------------
# CONCEPT DRIFT: a relação entre as features e o target mudou? (o modelo
# passou a errar mais, mesmo que os dados de entrada pareçam parecidos —
# isso é o que 01_drift_dados.py NÃO consegue ver).
#
# Limitação real deste dataset: TARGET_COL tem horizonte de 2 anos
# (SeriousDlqin2yrs = inadimplência nos PRÓXIMOS 24 meses). Isso significa
# que o resultado real de um cliente pontuado hoje só é conhecido daqui a
# ~2 anos — não dá pra medir concept drift de verdade sem esperar o rótulo
# atrasado chegar. Duas fontes de dado, portanto:
#
#   REAL_MODE  = True   -> espera uma tabela com o resultado REALIZADO dos
#                          clientes pontuados no passado (ver seção 1B).
#                          Ainda não existe no projeto — este notebook detecta
#                          a ausência e cai para simulação automaticamente.
#   SIMULATION_MODE      -> gera um lote sintético com uma relação
#                          feature->target deliberadamente diferente da de
#                          treino, para testar se a lógica de alerta
#                          funciona ANTES de depender de rótulos reais.
#                          Lógica validada em conversas anteriores deste
#                          projeto (bugs de referência/baseline já corrigidos).
# =============================================================================

# COMMAND ----------

# =========================
# 0. CONFIGURAÇÃO
# =========================
import logging
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.text(
    "simulation_mode", "true"
)  # "false" quando a tabela de rótulos atrasados existir
CATALOG = dbutils.widgets.get("catalog")
SIMULATION_MODE = dbutils.widgets.get("simulation_mode").lower() == "true"

SCHEMA = "gold"
TARGET_COL = "target_dlq_2yrs"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.credito_risco_xgb"
OUTPUT_TABLE = f"{CATALOG}.ml.monitoramento_concept_drift"
MONITORING_EXPERIMENT = "/Shared/credito_risco_monitoramento"
# Tabela que ainda não existe no projeto — quando o processo de reconciliação
# de resultados reais for criado, ela deve ter (customer_id, resultado_realizado).
TABELA_RESULTADOS_REALIZADOS = f"{CATALOG}.{SCHEMA}.resultados_realizados"

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

QUEDA_AUC_ALERTA = 0.03  # queda absoluta de AUC-ROC vs. baseline que dispara alerta
N_SIMULADO = 20_000

# Curva de sensibilidade (só em SIMULATION_MODE): cada valor gera um lote com
# drift daquela intensidade (0 = sem drift, 1 = sinais comportamentais deixam
# de explicar o alvo). Responde "a partir de que tamanho de mudança o monitor
# dispara?" e, com 0.0, confirma que ele não dispara à toa.
INTENSIDADES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

mlflow.set_experiment(MONITORING_EXPERIMENT)

# O schema `ml` não é criado por nenhuma camada anterior do pipeline
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.ml")

# COMMAND ----------

# =========================
# 0b. IMPORT DA LÓGICA TESTÁVEL (src/) — coberta por tests/test_concept_drift.py
# =========================
import os
import sys


def _find_repo_root(start: str) -> str:
    path = start
    for _ in range(6):
        if os.path.isdir(os.path.join(path, "src")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    raise RuntimeError(f"Não encontrei a pasta 'src' subindo a partir de {start}.")


repo_root = _find_repo_root(os.getcwd())
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.monitoring.concept_drift import avaliar_queda_auc, gerar_lote_sintetico

# COMMAND ----------

# =========================
# 1. CARREGA O @champion
# =========================
client = MlflowClient()
versao_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
modelo = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
print(f"Champion carregado: {MODEL_NAME} versão {versao_champion.version}")

# COMMAND ----------

# =========================
# 1B. TENTA MODO REAL — cai para simulação se a tabela de rótulos não existir
# =========================
df_avaliacao = None
auc_referencia = None

if not SIMULATION_MODE:
    try:
        df_avaliacao = spark.table(TABELA_RESULTADOS_REALIZADOS).toPandas()
        print(
            f"[REAL_MODE] {len(df_avaliacao)} resultados realizados carregados de {TABELA_RESULTADOS_REALIZADOS}."
        )
        # AUC de referência = a do próprio treino, registrada como tag no momento do registro do modelo.
        auc_referencia = float(
            client.get_model_version(MODEL_NAME, versao_champion.version).tags.get("auc_roc", "nan")
        )
        if np.isnan(auc_referencia):
            print(
                "[AVISO] Tag 'auc_roc' não encontrada na versão do modelo — "
                "adicione client.set_model_version_tag(..., 'auc_roc', ...) no notebook de treino."
            )
    except Exception as e:
        print(
            f"[AVISO] {TABELA_RESULTADOS_REALIZADOS} não existe ainda ({e}). "
            f"Sem rótulos reais atrasados, não dá pra medir concept drift de verdade. "
            f"Caindo para SIMULATION_MODE para validar a lógica de alerta."
        )
        SIMULATION_MODE = True

# COMMAND ----------

# =========================
# 2. REFERÊNCIA — depende do modo (gerar_lote_sintetico vem de src/)
# =========================
if SIMULATION_MODE:
    df_referencia_sim = gerar_lote_sintetico(N_SIMULADO, intensidade=0.0, seed=1)
    proba_referencia = modelo.predict_proba(df_referencia_sim[FEATURE_COLS])[:, 1]
    auc_referencia = float(roc_auc_score(df_referencia_sim[TARGET_COL], proba_referencia))

    print(f"[SIMULATION_MODE] Referência sintética (sem drift): AUC-ROC = {auc_referencia:.4f}")
    print(
        "Essa AUC não é comparável 1:1 ao baseline real de treino — os dados são "
        "sintéticos. O que importa é a QUEDA relativa entre referência e cada lote."
    )

# COMMAND ----------

# =========================
# 3. AVALIA OS LOTES CONTRA A REFERÊNCIA
# =========================
def avaliar_lote(df: pd.DataFrame, intensidade) -> dict:
    proba = modelo.predict_proba(df[FEATURE_COLS])[:, 1]
    y = df[TARGET_COL].to_numpy()
    auc = float(roc_auc_score(y, proba))
    ks = float(ks_2samp(proba[y == 1], proba[y == 0]).statistic)
    avaliacao = avaliar_queda_auc(auc_referencia, auc, limiar=QUEDA_AUC_ALERTA)
    return {
        "intensidade": intensidade,
        "auc_referencia": auc_referencia,
        "auc_atual": auc,
        "queda_auc": avaliacao["queda_auc"],
        "ks_atual": ks,
        "alerta": bool(avaliacao["alerta"]),
    }


if SIMULATION_MODE:
    resultados = [
        avaliar_lote(gerar_lote_sintetico(N_SIMULADO, intensidade=k, seed=2 + i), k)
        for i, k in enumerate(INTENSIDADES)
    ]
else:
    resultados = [avaliar_lote(df_avaliacao, None)]

df_resultado = pd.DataFrame(resultados)
print(f"AUC-ROC referência: {auc_referencia:.4f} | limiar de alerta: queda >= {QUEDA_AUC_ALERTA}\n")
print(df_resultado[["intensidade", "auc_atual", "queda_auc", "ks_atual", "alerta"]].to_string(index=False))

com_alerta = df_resultado[df_resultado["alerta"]]
menor_intensidade_alerta = (
    float(com_alerta["intensidade"].min())
    if SIMULATION_MODE and not com_alerta.empty
    else None
)

if SIMULATION_MODE:
    if menor_intensidade_alerta is None:
        print("\n[ATENÇÃO] Nenhuma intensidade disparou o alerta — revisar o gerador ou o limiar.")
    else:
        print(f"\nO monitor passa a disparar a partir da intensidade {menor_intensidade_alerta:.2f}.")
    sem_drift = df_resultado[df_resultado["intensidade"] == 0.0]
    if not sem_drift.empty and bool(sem_drift["alerta"].iloc[0]):
        print("[ATENÇÃO] Alerta disparou SEM drift (intensidade 0) — falso alarme.")
elif df_resultado["alerta"].iloc[0]:
    print("\n[ALERTA] CONCEPT DRIFT — a relação entre features e alvo parece ter mudado. Considere retreino.")
else:
    print("\nQueda de AUC-ROC dentro do esperado. Sem alerta de concept drift.")

# COMMAND ----------

# =========================
# 3B. GRÁFICO DA CURVA DE SENSIBILIDADE (só simulação)
# =========================
fig = None
if SIMULATION_MODE:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df_resultado["intensidade"], df_resultado["queda_auc"], marker="o", color="#1f4e79")
    ax.axhline(QUEDA_AUC_ALERTA, color="#c0392b", linestyle="--", label=f"Limiar de alerta ({QUEDA_AUC_ALERTA})")
    for _, linha in df_resultado.iterrows():
        ax.annotate(f"{linha['queda_auc']:.3f}", (linha["intensidade"], linha["queda_auc"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Intensidade do concept drift simulado (0 = sem drift, 1 = drift total)")
    ax.set_ylabel("Queda de AUC-ROC vs. referência")
    ax.set_title("Sensibilidade do monitor de concept drift")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    display(fig)

# COMMAND ----------

# =========================
# 4. LOGA NO MLFLOW E PERSISTE HISTÓRICO
# =========================
with mlflow.start_run(run_name="concept_drift"):
    mlflow.log_param("simulation_mode", SIMULATION_MODE)
    mlflow.log_param("limiar_queda_auc", QUEDA_AUC_ALERTA)
    mlflow.log_metric("auc_referencia", auc_referencia)
    for _, linha in df_resultado.iterrows():
        # step = intensidade x 100 -> o MLflow desenha a curva de sensibilidade
        step = int(round(linha["intensidade"] * 100)) if SIMULATION_MODE else 0
        mlflow.log_metric("auc_atual", linha["auc_atual"], step=step)
        mlflow.log_metric("queda_auc", linha["queda_auc"], step=step)
        mlflow.log_metric("ks_atual", linha["ks_atual"], step=step)
        mlflow.log_metric("alerta_concept_drift", int(linha["alerta"]), step=step)
    if menor_intensidade_alerta is not None:
        mlflow.log_metric("menor_intensidade_alerta", menor_intensidade_alerta)
    if fig is not None:
        mlflow.log_figure(fig, "concept_drift_sensibilidade.png")

df_persistir = df_resultado.copy()
df_persistir["timestamp_execucao"] = datetime.now(timezone.utc)
df_persistir["simulation_mode"] = SIMULATION_MODE
df_persistir["intensidade"] = df_persistir["intensidade"].astype("float64")

# mergeSchema: a coluna `intensidade` não existia nas execuções anteriores
(
    spark.createDataFrame(df_persistir)
    .write.mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"\nResultado gravado em {OUTPUT_TABLE} e logado no experimento '{MONITORING_EXPERIMENT}'.")
