# Databricks notebook source
# ==============================================================================
# notebooks/ingestao/01_bronze.py
# Orquestração da ingestão Bronze — lógica real vive em src/ingestion/bronze.py
# ==============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros de execução
# MAGIC O catálogo é injetado via widget para permitir promoção dev → hml → prd
# MAGIC sem alterar código (Databricks Asset Bundles / Jobs).

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import do pacote `src`
# MAGIC Em um Databricks Repo (Git folder), a raiz do repositório já entra no
# MAGIC `sys.path` automaticamente. O `sys.path.append` abaixo é apenas um
# MAGIC fallback defensivo caso o notebook seja movido para fora da estrutura
# MAGIC padrão do repo.

# COMMAND ----------

import sys
import os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.ingestion.bronze import run_bronze_ingestion

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execução

# COMMAND ----------

run_bronze_ingestion(spark, catalog=catalog)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checagem rápida pós-carga

# COMMAND ----------

display(spark.table(f"{catalog}.bronze.give_me_some_credit_raw").limit(10))