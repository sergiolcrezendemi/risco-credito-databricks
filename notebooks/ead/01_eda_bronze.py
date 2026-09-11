# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
%uv sync

# COMMAND ----------

# ==============================================================================
# notebooks/ead/01_eda_bronze.py
# EDA da camada Bronze — lê a tabela já ingerida por 01_bronze.py. Nenhuma
# lógica de transformação aqui, só diagnóstico (src/quality/profiling.py).
# ==============================================================================

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.dropdown("dataset", "training", ["training", "scoring"])

catalog = dbutils.widgets.get("catalog")
dataset = dbutils.widgets.get("dataset")

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

# COMMAND ----------

table_name = f"{catalog}.bronze.give_me_some_credit_{dataset}_raw"
df_bronze = spark.table(table_name)

numeric_cols = [
    "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

df_diagnostico_nulos = run_profile(
    df_bronze,
    table_label=table_name,
    key_cols=["customer_id"],
    numeric_cols=numeric_cols,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualização gráfica da completude por coluna

# COMMAND ----------

display(df_diagnostico_nulos)