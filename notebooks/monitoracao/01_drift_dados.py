# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# =============================================================================
# notebooks/monitoracao/01_drift_dados.py
# -----------------------------------------------------------------------------
# DATA DRIFT: a distribuição das features de ENTRADA mudou em relação à base
# que treinou o @champion? Isso é independente de saber se o modelo ainda
# acerta (isso é concept drift, tratado em 02_concept_drift.py) — aqui a
# pergunta é só "os dados que estão chegando parecem com os dados de treino?"
#
# Separação treino x scoring no Gold:
#   treino  = registros com TARGET_COL preenchido (origem cs-training.csv)
#   scoring = registros com TARGET_COL nulo       (origem cs-test.csv)
# Mesmo critério usado em 03_inferencia_batch.py.
#
# Widget `base_atual` — o que é comparado contra a REFERÊNCIA (treino):
#   scoring          -> lote de scoring. Uso normal: mede drift de verdade.
#   training         -> o próprio treino contra ele mesmo. Teste de sanidade:
#                       PSI tem que ser 0 em todas as features. Se não for,
#                       há bug na carga ou no cálculo.
#   training_amostra -> 20% do treino (hash determinístico de customer_id)
#                       contra os outros 80%. Teste realista: PSI deve ficar
#                       muito baixo (estável), sem chegar a zero.
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
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.getLogger("mlflow").setLevel(logging.ERROR)

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.dropdown("base_atual", "scoring", ["scoring", "training", "training_amostra"])
CATALOG = dbutils.widgets.get("catalog")
BASE_ATUAL = dbutils.widgets.get("base_atual")

SCHEMA = "gold"
FCT_TABLE = "fct_credit_profile"
DIM_CUSTOMER_TABLE = "dim_customer"
TARGET_COL = "target_dlq_2yrs"
OUTPUT_TABLE = f"{CATALOG}.ml.monitoramento_drift_dados"
MONITORING_EXPERIMENT = "/Shared/credito_risco_monitoramento"

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

PSI_MODERADO = 0.10
PSI_SEVERO = 0.25
PCT_AMOSTRA = 20  # % do treino separado em base_atual=training_amostra

mlflow.set_experiment(MONITORING_EXPERIMENT)

# COMMAND ----------

# =========================
# 0b. IMPORT DA LÓGICA TESTÁVEL (src/) — coberta por tests/test_data_drift.py
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

from src.monitoring.data_drift import calcular_psi, classificar_psi

# COMMAND ----------

# =========================
# 1. CARREGA REFERÊNCIA E LOTE ATUAL CONFORME `base_atual`
# =========================
# pmod garante bucket 0..99 mesmo quando o hash é negativo.
FILTRO_TREINO = f"f.{TARGET_COL} IS NOT NULL"
FILTRO_SCORING = f"f.{TARGET_COL} IS NULL"
BUCKET = "pmod(xxhash64(f.customer_id), 100)"

if BASE_ATUAL == "scoring":
    filtro_referencia, filtro_atual = FILTRO_TREINO, FILTRO_SCORING
    descricao = "referência = treino | atual = scoring"
elif BASE_ATUAL == "training":
    filtro_referencia, filtro_atual = FILTRO_TREINO, FILTRO_TREINO
    descricao = "referência = treino | atual = treino (teste de sanidade: PSI esperado = 0)"
else:  # training_amostra
    filtro_referencia = f"{FILTRO_TREINO} AND {BUCKET} >= {PCT_AMOSTRA}"
    filtro_atual = f"{FILTRO_TREINO} AND {BUCKET} < {PCT_AMOSTRA}"
    descricao = (
        f"referência = {100 - PCT_AMOSTRA}% do treino | atual = {PCT_AMOSTRA}% do treino "
        "(PSI esperado muito baixo)"
    )


def carregar(filtro_sql: str) -> pd.DataFrame:
    df_spark = spark.sql(f"""
        SELECT c.age, c.num_dependents, f.monthly_income, f.debt_ratio, f.revolving_utilization,
               f.num_open_credit_lines, f.num_real_estate_loans, f.num_times_30_59_days_late,
               f.num_times_60_89_days_late, f.num_times_90_days_late, f.total_delinquency_events
        FROM {CATALOG}.{SCHEMA}.{FCT_TABLE} f
        JOIN {CATALOG}.{SCHEMA}.{DIM_CUSTOMER_TABLE} c ON f.customer_id = c.customer_id
        WHERE {filtro_sql}
    """)
    return df_spark.toPandas()


df_referencia = carregar(filtro_referencia)
df_atual = carregar(filtro_atual)

if df_referencia.empty or df_atual.empty:
    # RuntimeError (e não dbutils.notebook.exit) para o Job FALHAR e não
    # aparecer como "Succeeded" sem ter monitorado nada.
    raise RuntimeError(
        f"[base_atual={BASE_ATUAL}] Referência ({len(df_referencia)} linhas) ou lote atual "
        f"({len(df_atual)} linhas) vazios — nada a comparar. "
        "Se o lote atual é o scoring, rode antes o job_ingestao_scoring "
        "(a Gold precisa ter linhas com target_dlq_2yrs nulo)."
    )

print(f"Modo: {BASE_ATUAL} -> {descricao}")
print(f"Referência: {len(df_referencia)} linhas | Lote atual: {len(df_atual)} linhas")

# COMMAND ----------

# =========================
# 2. CALCULA PSI + KS PARA CADA FEATURE
# =========================
linhas_resultado = []
alertas_severos = []

for feature in FEATURE_COLS:
    ref_vals = df_referencia[feature].to_numpy(dtype=float)
    atual_vals = df_atual[feature].to_numpy(dtype=float)

    psi = calcular_psi(ref_vals, atual_vals)
    classificacao = classificar_psi(psi, moderado=PSI_MODERADO, severo=PSI_SEVERO)

    ref_validos = ref_vals[~np.isnan(ref_vals)]
    atual_validos = atual_vals[~np.isnan(atual_vals)]
    ks_stat, ks_pvalue = (
        ks_2samp(ref_validos, atual_validos)
        if len(ref_validos) and len(atual_validos)
        else (float("nan"), float("nan"))
    )

    linhas_resultado.append(
        {
            "feature": feature,
            "psi": psi,
            "classificacao_psi": classificacao,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
        }
    )
    if classificacao == "SEVERO":
        alertas_severos.append(feature)

df_drift = pd.DataFrame(linhas_resultado).sort_values("psi", ascending=False)
print(df_drift.to_string(index=False))

if alertas_severos:
    print(f"\n[ALERTA] DATA DRIFT SEVERO nas features: {alertas_severos}")
    print("Considere investigar a origem dos dados antes de confiar nas previsões do @champion.")
else:
    print("\nNenhuma feature com drift severo neste lote.")

# Leitura esperada por modo — ajuda a validar o próprio notebook
psi_max = float(df_drift["psi"].max())
if BASE_ATUAL == "training":
    print(
        f"\n[TESTE DE SANIDADE] PSI máximo = {psi_max:.6f} "
        f"({'OK' if psi_max < 1e-9 else 'FALHOU — deveria ser 0'})"
    )
elif BASE_ATUAL == "training_amostra":
    print(
        f"\n[TESTE REALISTA] PSI máximo = {psi_max:.4f} "
        f"({'OK, estável' if psi_max < PSI_MODERADO else 'ATENÇÃO — amostra do mesmo treino não deveria ter drift'})"
    )

# COMMAND ----------

# =========================
# 3. LOGA NO MLFLOW E PERSISTE HISTÓRICO
# =========================
with mlflow.start_run(run_name=f"drift_dados_{BASE_ATUAL}"):
    mlflow.log_param("base_atual", BASE_ATUAL)
    mlflow.log_param("n_referencia", len(df_referencia))
    mlflow.log_param("n_atual", len(df_atual))
    for _, linha in df_drift.iterrows():
        mlflow.log_metric(f"psi_{linha['feature']}", linha["psi"])
        mlflow.log_metric(f"ks_{linha['feature']}", linha["ks_statistic"])
    mlflow.log_metric("qtd_features_drift_severo", len(alertas_severos))

df_drift["timestamp_execucao"] = datetime.now(timezone.utc)
df_drift["base_atual"] = BASE_ATUAL
df_drift["n_referencia"] = len(df_referencia)
df_drift["n_atual"] = len(df_atual)


def garantir_colunas(tabela: str, colunas: dict) -> None:
    """Se a tabela já existe, adiciona as colunas que faltam (ALTER TABLE).
    Se ainda não existe, o saveAsTable abaixo a cria já com todas."""
    if not spark.catalog.tableExists(tabela):
        return
    existentes = {c.lower() for c in spark.table(tabela).columns}
    faltantes = {n: t for n, t in colunas.items() if n.lower() not in existentes}
    if faltantes:
        defs = ", ".join(f"`{n}` {t}" for n, t in faltantes.items())
        spark.sql(f"ALTER TABLE {tabela} ADD COLUMNS ({defs})")
        print(f"[schema] Colunas adicionadas em {tabela}: {list(faltantes)}")


garantir_colunas(OUTPUT_TABLE, {"base_atual": "STRING"})

# mergeSchema fica como segunda garantia caso surja outra coluna no futuro.
(
    spark.createDataFrame(df_drift)
    .write.mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"\nResultado gravado em {OUTPUT_TABLE} e logado no experimento '{MONITORING_EXPERIMENT}'.")

