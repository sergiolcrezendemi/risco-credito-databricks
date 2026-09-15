# Databricks notebook source
%python
# Databricks notebook source
# =============================================================================
# notebooks/ml/03_inferencia_batch.py
# -----------------------------------------------------------------------------
# Fase 6 do roteiro-execucao.md — "entrada de novos dados": pega o dataset
# scoring (holdout, sem target — já ingerido pela Fase 1/01_bronze.py com
# dataset=scoring e tratado pelas Fases 2/3), aplica o modelo @champion
# registrado no Unity Catalog Model Registry, e persiste probabilidade +
# decisão de aprovação por cliente.
#
# É o passo que alimenta os dois notebooks de monitoramento que vêm depois
# (01_drift_dados.py e 02_concept_drift.py) — ambos leem o resultado desta
# tabela ou os mesmos dados de origem que ela usou.
#
# Convenção sobre "quem é o dataset scoring": no Gold (fct_credit_profile),
# assumimos que os registros de scoring são os que têm TARGET_COL nulo —
# é assim que a Bronze os ingeriu (o CSV de scoring não tem a coluna
# SeriousDlqin2yrs). Se no seu Silver/Gold real a separação for por tabela
# ou por uma coluna explícita (ex.: `dataset_origem`), ajuste o WHERE da
# seção 2 — o resto do notebook não muda.
# =============================================================================

# COMMAND ----------
# =========================
# 0. CONFIGURAÇÃO
# =========================
import logging
import mlflow
import mlflow.xgboost
import pandas as pd
from datetime import datetime, timezone
from mlflow.tracking import MlflowClient

logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.text("threshold_aprovacao", "0.56")  # Q2 do README — ajustar se src/config/business_params.py já centralizar isso
CATALOG = dbutils.widgets.get("catalog")
THRESHOLD_APROVACAO = float(dbutils.widgets.get("threshold_aprovacao"))

SCHEMA = "gold"
FCT_TABLE = "fct_credit_profile"
DIM_CUSTOMER_TABLE = "dim_customer"
TARGET_COL = "target_dlq_2yrs"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.credito_risco_xgb"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.credito_score_predictions"

FEATURE_COLS = [
    "age", "num_dependents", "monthly_income", "debt_ratio", "revolving_utilization",
    "num_open_credit_lines", "num_real_estate_loans", "num_times_30_59_days_late",
    "num_times_60_89_days_late", "num_times_90_days_late", "total_delinquency_events",
]

# COMMAND ----------
# =========================
# 0b. IMPORT DA LÓGICA TESTÁVEL (src/) — coberta por tests/test_decisao.py
# =========================
import sys, os
sys.path.append(os.path.abspath("../../src"))  # ajustar se a profundidade do repo for diferente
from risco_credito.ml.decisao import classificar_decisao

# COMMAND ----------
# =========================
# 1. CARREGA O MODELO @champion
# =========================
client = MlflowClient()
try:
    versao_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    modelo = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
    print(f"Champion carregado: {MODEL_NAME} versão {versao_champion.version}")
except Exception as e:
    raise RuntimeError(
        f"Não foi possível carregar '{MODEL_NAME}@champion'. Rode antes o notebook "
        f"de treino (02_vies_threshold_roi.py) para registrar um @champion. Erro original: {e}"
    )

# Nota: carregamos com mlflow.xgboost.load_model (não mlflow.pyfunc) de propósito —
# o flavor pyfunc, dependendo da versão do MLflow, pode devolver classe (0/1) em vez
# de probabilidade em .predict(). Com o flavor nativo, .predict_proba()[:, 1] é
# sempre a probabilidade da classe positiva, sem ambiguidade.

# COMMAND ----------
# =========================
# 2. CARREGA O LOTE NOVO (dataset scoring — TARGET_COL nulo)
# =========================
df_spark = spark.sql(f"""
    SELECT
        f.customer_id,
        c.age, c.num_dependents, f.monthly_income, f.debt_ratio, f.revolving_utilization,
        f.num_open_credit_lines, f.num_real_estate_loans, f.num_times_30_59_days_late,
        f.num_times_60_89_days_late, f.num_times_90_days_late, f.total_delinquency_events
    FROM {CATALOG}.{SCHEMA}.{FCT_TABLE} f
    JOIN {CATALOG}.{SCHEMA}.{DIM_CUSTOMER_TABLE} c ON f.customer_id = c.customer_id
    WHERE f.{TARGET_COL} IS NULL
""")
df = df_spark.toPandas()

if df.empty:
    dbutils.notebook.exit("[AVISO] Nenhum registro novo (TARGET_COL nulo) encontrado para pontuar. Encerrando.")

print(f"Registros a pontuar: {len(df)}")

nulos_por_coluna = df[FEATURE_COLS].isna().sum()
nulos_por_coluna = nulos_por_coluna[nulos_por_coluna > 0]
if not nulos_por_coluna.empty:
    print("[AVISO] Colunas com nulos no lote novo (XGBoost lida nativamente, mas vale checar a origem):")
    for col, qtd in nulos_por_coluna.items():
        print(f"  {col}: {qtd} ({qtd / len(df):.2%})")

# COMMAND ----------
# =========================
# 3. SCORING
# =========================
probabilidades = modelo.predict_proba(df[FEATURE_COLS])[:, 1]

df_resultado = pd.DataFrame({
    "customer_id": df["customer_id"],
    "probabilidade_inadimplencia": probabilidades,
    "decisao": [classificar_decisao(p, THRESHOLD_APROVACAO) for p in probabilidades],
    "modelo_nome": MODEL_NAME,
    "modelo_versao": versao_champion.version,
    "threshold_usado": THRESHOLD_APROVACAO,
    "timestamp_execucao": datetime.now(timezone.utc),
})

taxa_negacao = (df_resultado["decisao"] == "negado").mean()
print(f"Threshold de aprovação: {THRESHOLD_APROVACAO}")
print(f"Taxa de negação neste lote: {taxa_negacao:.2%}")
print(df_resultado["probabilidade_inadimplencia"].describe())

# COMMAND ----------
# =========================
# 4. PERSISTE O RESULTADO (append — cada execução soma um novo lote com seu timestamp)
# =========================
spark.createDataFrame(df_resultado).write.mode("append").saveAsTable(OUTPUT_TABLE)
print(f"\n{len(df_resultado)} previsões gravadas em {OUTPUT_TABLE}.")
print("Inferência em lote concluída.")
