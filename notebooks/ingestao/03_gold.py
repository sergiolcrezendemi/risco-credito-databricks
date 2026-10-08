# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %%sh
# MAGIC # IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
# MAGIC uv sync

# COMMAND ----------

# ==============================================================================
# notebooks/ingestao/03_gold.py
# Orquestração da construção da Gold — lógica real vive em src/features/gold.py
# ==============================================================================

"""
### Parâmetros de execução
Constrói `gold.dim_customer` e `gold.fct_credit_profile` a partir da UNIÃO de
`silver.give_me_some_credit` (treino) com `silver.give_me_some_credit_scoring`
(holdout, se existir). Linhas de scoring ficam com `target_dlq_2yrs` nulo por
construção — é assim que `notebooks/ml/03_inferencia_batch.py` e os notebooks
de `notebooks/monitoracao/` identificam o lote novo a pontuar. Rode depois de
`02_silver.py` (`dataset=training` e, se aplicável, `dataset=scoring`) já
terem populado a Silver.
"""

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

import sys
import os


def _find_repo_root(start: str) -> str:
    path = start
    for _ in range(6):  # limite de segurança
        if os.path.isdir(os.path.join(path, "src")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    raise RuntimeError(
        f"Não encontrei a pasta 'src' subindo a partir de {start}. "
        f"Confirme se este notebook está dentro da estrutura do repositório."
    )


repo_root = _find_repo_root(os.getcwd())
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.features.gold import run_gold_ingestion

# %md
### Execução

# %%
run_gold_ingestion(spark, catalog=catalog)

# %md
# ### Checagem rápida pós-carga

# %%
display(spark.table(f"{catalog}.gold.dim_customer").limit(10))

# %%
display(spark.table(f"{catalog}.gold.fct_credit_profile").limit(10))
