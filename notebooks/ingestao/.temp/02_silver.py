# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# ==============================================================================
# notebooks/ingestao/02_silver.py
# Orquestração da ingestão Silver — lógica real vive em src/ingestion/silver.py
# ==============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros de execução

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import do pacote `src`

# COMMAND ----------

import sys
import os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.ingestion.silver import run_silver_ingestion

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execução
# MAGIC O `run_silver_ingestion` já garante o volume de checkpoint antes de
# MAGIC iniciar o `writeStream` (corrige o UC_VOLUME_NOT_FOUND observado
# MAGIC anteriormente) e aplica o schema único de `src/ingestion/silver.py`.

# COMMAND ----------

run_silver_ingestion(spark, catalog=catalog)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checagem rápida pós-carga

# COMMAND ----------

display(spark.table(f"{catalog}.silver.give_me_some_credit").limit(10))