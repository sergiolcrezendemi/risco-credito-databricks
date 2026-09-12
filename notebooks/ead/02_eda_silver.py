# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
%uv sync

# COMMAND ----------

# ==============================================================================
# notebooks/ead/02_eda_silver.py
# EDA da camada Silver — lê `silver.give_me_some_credit`, já construída por
# 02_silver.py. Corrige o padrão da versão original (EAD_silver.ipynb), que
# recalculava toda a limpeza Bronze->Silver dentro do próprio notebook de
# EDA — duplicando a lógica de negócio que já vive em src/ingestion/silver.py
# e arriscando divergir dela com o tempo.
# ==============================================================================

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

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

from src.quality.profiling import run_profile
from pyspark.sql import functions as F

# COMMAND ----------

table_name = f"{catalog}.silver.give_me_some_credit"
df_silver = spark.table(table_name)

numeric_cols = [
    "age", "debt_ratio", "monthly_income", "revolving_utilization_unsecured",
    "num_open_credit_lines_and_loans", "num_times_30_59_days_late",
    "num_times_60_89_days_late", "num_times_90_days_late",
    "num_real_estate_loans_or_lines", "num_dependents",
]

df_diagnostico_nulos = run_profile(
    df_silver,
    table_label=table_name,
    key_cols=["customer_id"],
    numeric_cols=numeric_cols,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribuição da variável-alvo (`target_default_2yrs`)

# COMMAND ----------

display(
    df_silver.groupBy("target_default_2yrs")
    .count()
    .withColumn("percentual", F.round((F.col("count") / df_silver.count()) * 100, 2))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mesmo gráfico, via matplotlib (caso prefira renderização estática)

# COMMAND ----------

import matplotlib.pyplot as plt

pdf_target = (
    df_silver
    .withColumn(
        "status_cliente",
        F.when(F.col("target_default_2yrs") == 1, "Inadimplente (1)").otherwise("Adimplente (0)"),
    )
    .groupBy("status_cliente")
    .count()
    .toPandas()
)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(pdf_target["status_cliente"], pdf_target["count"], color=["#1f77b4", "#d62728"])
ax.bar_label(bars, fmt="{:,.0f}", label_type="center", color="white", fontweight="bold", fontsize=11)
ax.set_title("Distribuição do Target", fontsize=12, fontweight="bold")
ax.set_xlabel("Status do Cliente")
ax.set_ylabel("Contagem")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checagem de qualidade (flags herdadas da Silver)

# COMMAND ----------

display(
    df_silver.select(
        F.sum("is_monthly_income_null").alias("qtd_renda_nula"),
        F.sum("has_delinquency_outlier").alias("qtd_outlier_atraso"),
        F.sum(F.when(F.col("age").isNull(), 1).otherwise(0)).alias("qtd_idade_em_quarentena"),
    )
)