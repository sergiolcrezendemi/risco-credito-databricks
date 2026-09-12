# Databricks notebook source
%python
# Databricks notebook source
# =============================================================================
# notebooks/monitoracao/01_drift_dados.py
# -----------------------------------------------------------------------------
# DATA DRIFT: a distribuição das features de ENTRADA mudou em relação à base
# que treinou o @champion? Isso é independente de saber se o modelo ainda
# acerta (isso é concept drift, tratado em 02_concept_drift.py) — aqui a
# pergunta é só "os dados que estão chegando parecem com os dados de treino?"
#
# Referência (treino) = registros do Gold com TARGET_COL preenchido.
# Lote atual (scoring) = registros do Gold com TARGET_COL nulo — o mesmo
# filtro usado em 03_inferencia_batch.py. Ajuste os dois SELECTs abaixo se
# no seu ambiente real essa separação for por tabela em vez de coluna nula.
#
# Métrica principal: PSI (Population Stability Index) por feature, com
# KS-statistic como segunda leitura. Limiares (documentados, ajustar com a
# área de risco):
#   PSI < 0.10            -> estável
#   0.10 <= PSI < 0.25     -> moderado (observar)
#   PSI >= 0.25            -> severo (investigar antes de confiar no @champion)
# =============================================================================

# COMMAND ----------
# =========================
# 0. CONFIGURAÇÃO
# =========================
import logging
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime, timezone
from scipy.stats import ks_2samp

logging.getLogger("mlflow").setLevel(logging.ERROR)

dbutils.widgets.text("catalog", "credito_dev")
CATALOG = dbutils.widgets.get("catalog")

SCHEMA = "gold"
FCT_TABLE = "fct_credit_profile"
DIM_CUSTOMER_TABLE = "dim_customer"
TARGET_COL = "target_dlq_2yrs"
OUTPUT_TABLE = f"{CATALOG}.ml.monitoramento_drift_dados"
MONITORING_EXPERIMENT = "/Shared/credito_risco_monitoramento"

FEATURE_COLS = [
    "age", "num_dependents", "monthly_income", "debt_ratio", "revolving_utilization",
    "num_open_credit_lines", "num_real_estate_loans", "num_times_30_59_days_late",
    "num_times_60_89_days_late", "num_times_90_days_late", "total_delinquency_events",
]

PSI_MODERADO = 0.10
PSI_SEVERO = 0.25

mlflow.set_experiment(MONITORING_EXPERIMENT)

# COMMAND ----------
# =========================
# 0b. IMPORT DA LÓGICA TESTÁVEL (src/) — coberta por tests/test_data_drift.py
# =========================
import sys
import os


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

from src.monitoring.data_drift import calcular_psi, classificar_psi

# COMMAND ----------
# =========================
# 1. CARREGA REFERÊNCIA (treino) E LOTE ATUAL (scoring)
# =========================
def carregar(where_target_nulo: bool) -> pd.DataFrame:
    condicao = "IS NULL" if where_target_nulo else "IS NOT NULL"
    df_spark = spark.sql(f"""
        SELECT c.age, c.num_dependents, f.monthly_income, f.debt_ratio, f.revolving_utilization,
               f.num_open_credit_lines, f.num_real_estate_loans, f.num_times_30_59_days_late,
               f.num_times_60_89_days_late, f.num_times_90_days_late, f.total_delinquency_events
        FROM {CATALOG}.{SCHEMA}.{FCT_TABLE} f
        JOIN {CATALOG}.{SCHEMA}.{DIM_CUSTOMER_TABLE} c ON f.customer_id = c.customer_id
        WHERE f.{TARGET_COL} {condicao}
    """)
    return df_spark.toPandas()

df_referencia = carregar(where_target_nulo=False)
df_atual = carregar(where_target_nulo=True)

if df_referencia.empty or df_atual.empty:
    dbutils.notebook.exit(
        f"[AVISO] Referência ({len(df_referencia)} linhas) ou lote atual ({len(df_atual)} linhas) "
        f"vazios — nada a comparar. Rode 03_inferencia_batch.py primeiro se o lote atual estiver vazio."
    )

print(f"Referência (treino): {len(df_referencia)} linhas | Lote atual (scoring): {len(df_atual)} linhas")

# COMMAND ----------
# =========================
# 3. CALCULA PSI + KS PARA CADA FEATURE
# =========================
linhas_resultado = []
alertas_severos = []

for feature in FEATURE_COLS:
    ref_vals = df_referencia[feature].to_numpy(dtype=float)
    atual_vals = df_atual[feature].to_numpy(dtype=float)

    psi = calcular_psi(ref_vals, atual_vals)
    classificacao = classificar_psi(psi)

    ref_validos = ref_vals[~np.isnan(ref_vals)]
    atual_validos = atual_vals[~np.isnan(atual_vals)]
    ks_stat, ks_pvalue = (
        ks_2samp(ref_validos, atual_validos) if len(ref_validos) and len(atual_validos) else (float("nan"), float("nan"))
    )

    linhas_resultado.append({
        "feature": feature, "psi": psi, "classificacao_psi": classificacao,
        "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_pvalue),
    })
    if classificacao == "SEVERO":
        alertas_severos.append(feature)

df_drift = pd.DataFrame(linhas_resultado).sort_values("psi", ascending=False)
print(df_drift.to_string(index=False))

if alertas_severos:
    print(f"\n[ALERTA] DATA DRIFT SEVERO nas features: {alertas_severos}")
    print("Considere investigar a origem dos dados antes de confiar nas previsões do @champion.")
else:
    print("\nNenhuma feature com drift severo neste lote.")

# COMMAND ----------
# =========================
# 4. LOGA NO MLFLOW E PERSISTE HISTÓRICO
# =========================
with mlflow.start_run(run_name="drift_dados"):
    mlflow.log_param("n_referencia", len(df_referencia))
    mlflow.log_param("n_atual", len(df_atual))
    for _, linha in df_drift.iterrows():
        mlflow.log_metric(f"psi_{linha['feature']}", linha["psi"])
        mlflow.log_metric(f"ks_{linha['feature']}", linha["ks_statistic"])
    mlflow.log_metric("qtd_features_drift_severo", len(alertas_severos))

df_drift["timestamp_execucao"] = datetime.now(timezone.utc)
df_drift["n_referencia"] = len(df_referencia)
df_drift["n_atual"] = len(df_atual)
spark.createDataFrame(df_drift).write.mode("append").saveAsTable(OUTPUT_TABLE)

print(f"\nResultado gravado em {OUTPUT_TABLE} e logado no experimento '{MONITORING_EXPERIMENT}'.")
