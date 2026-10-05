# Databricks notebook source
# scripts/setup/seed_hml_volumes.py
# ==============================================================================
# Script de setup: popula os volumes de aterragem (landing) a partir de DEV
# ==============================================================================

# MAGIC %pip install databricks-sdk

# COMMAND ----------

dbutils.widgets.text("source_catalog", "credito_dev", "Catálogo de Origem")
dbutils.widgets.text("target_catalog", "credito_hml", "Catálogo de Destino")

source_catalog = dbutils.widgets.get("source_catalog")
target_catalog = dbutils.widgets.get("target_catalog")

datasets = ["training", "scoring"]

for ds in datasets:
    src_path = f"/Volumes/{source_catalog}/bronze/raw_landing/{ds}/"
    tgt_path = f"/Volumes/{target_catalog}/bronze/raw_landing/{ds}/"

    print(f"Copiando dados de {src_path} para {tgt_path}...")
    try:
        dbutils.fs.cp(src_path, tgt_path, recurse=True)
        print(f"[OK] Volume {ds} populado com sucesso em {target_catalog}.")
    except Exception as e:
        print(f"[ERRO] Falha ao copiar {ds}: {str(e)}")
