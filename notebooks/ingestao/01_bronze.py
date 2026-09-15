# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
%uv sync --active --link-mode=copy -q

# COMMAND ----------

# ==============================================================================
# notebooks/ingestao/01_bronze.py
# Orquestração da ingestão Bronze — lógica real vive em src/ingestion/bronze.py
# ==============================================================================

"""
IMPORTANTE

 RODAR COM DUAS OPÇÕES  (responder a pergunta no inicio do cabeçalho)

 1- DATASET = training
 2- DATASEET = scoring       

"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros de execução
# MAGIC O catálogo é injetado via widget para permitir promoção dev → hml → prd
# MAGIC sem alterar código (Databricks Asset Bundles / Jobs).
# MAGIC
# MAGIC O widget `dataset` aponta para a subpasta em `raw_landing/` a
# MAGIC processar — `training` (cs-training.csv, com target) ou `scoring`
# MAGIC (cs-test.csv, sem target). Como são semanticamente diferentes, cada
# MAGIC valor gera sua própria tabela Bronze; nunca leia os dois juntos com
# MAGIC um glob genérico. Para carregar ambos, rode este notebook duas vezes
# MAGIC (ex.: duas tasks no mesmo Job, cada uma com um valor de `dataset`).

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.dropdown("dataset", "training", ["training", "scoring"])

catalog = dbutils.widgets.get("catalog")
dataset = dbutils.widgets.get("dataset")

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

run_bronze_ingestion(spark, catalog=catalog, dataset=dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checagem rápida pós-carga

# COMMAND ----------

display(spark.table(f"{catalog}.bronze.give_me_some_credit_{dataset}_raw").limit(10))