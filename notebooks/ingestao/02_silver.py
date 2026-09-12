# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
%uv sync

# COMMAND ----------

# ==============================================================================
# notebooks/ingestao/02_silver.py
# Orquestração da ingestão Silver — lógica real vive em src/ingestion/silver.py
# ==============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros de execução
# MAGIC `dataset=training` (default) é o que alimenta o modelo — origem
# MAGIC `bronze.give_me_some_credit_training_raw`, destino
# MAGIC `silver.give_me_some_credit`. `scoring` processa o holdout sem label
# MAGIC e grava em `silver.give_me_some_credit_scoring` — para inferência,
# MAGIC não para treino.

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.dropdown("dataset", "training", ["training", "scoring"])

catalog = dbutils.widgets.get("catalog")
dataset = dbutils.widgets.get("dataset")

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

run_silver_ingestion(spark, catalog=catalog, dataset=dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checagem rápida pós-carga

# COMMAND ----------

target_table_name = "give_me_some_credit" if dataset == "training" else f"give_me_some_credit_{dataset}"
display(spark.table(f"{catalog}.silver.{target_table_name}").limit(10))