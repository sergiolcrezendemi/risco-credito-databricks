# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
%uv sync

# COMMAND ----------

# ==============================================================================
# notebooks/ingestao/00_reset_ambiente.py
# Reset único do estado gerado pela estrutura ANTIGA (raw_landing sem
# subpastas training/scoring). Rode ANTES de 01_bronze.py e 02_silver.py
# na primeira execução com o novo layout. Depois disso, não precisa rodar
# de novo — não faz parte do pipeline recorrente.
# ==============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ### O que este notebook apaga
# MAGIC - Tabela `bronze.give_me_some_credit_raw` (misturava cs-training.csv e cs-test.csv)
# MAGIC - Tabela `silver.give_me_some_credit` (derivada da Bronze contaminada acima)
# MAGIC - Checkpoints e schemaLocation antigos em Volumes, ligados a essas duas tabelas
# MAGIC
# MAGIC Se essas tabelas/pastas não existirem (ex.: catálogo novo), cada remoção
# MAGIC vira um no-op seguro — nada quebra.
# MAGIC
# MAGIC **Não apaga** `raw_landing/training/` nem `raw_landing/scoring/` — os
# MAGIC arquivos `.csv` originais ficam intactos.

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.text("confirmar", "")  # precisa digitar exatamente: RESETAR

catalog = dbutils.widgets.get("catalog")
confirmar_texto = dbutils.widgets.get("confirmar")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import do pacote `src`

# COMMAND ----------

import sys
import os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.ingestion.maintenance import reset_legacy_environment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execução
# MAGIC Para rodar de fato, preencha o widget `confirmar` com o texto
# MAGIC `RESETAR` antes de executar esta célula. Qualquer outro valor
# MAGIC interrompe sem apagar nada.

# COMMAND ----------

reset_legacy_environment(
    spark=spark,
    dbutils=dbutils,
    catalog=catalog,
    confirm=(confirmar_texto == "RESETAR"),
)

"""
IMPORTANTE
    VAI APARAECER UMA CAIXA NO TOPO DA TELA, COM O TEXTO:
        "RESETAR"?
    PRECISA DIGITAR EXATAMENTE RESETAR, SEM ESPAÇOS, SEM MAIÚSCULAS, SEM APOSTROFES E SEM PONTUAÇÃO
    DEPOIS EXECUTA A CELULA NOVAMENTE
    
"""