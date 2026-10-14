# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingestão Bronze via Autoloader
# MAGIC
# MAGIC **Case:** `risco-credito-databricks`
# MAGIC **Fontes:** `cs-training.csv` (rotulado, treino) e `cs-test.csv` (sem rótulo — usado depois
# MAGIC como lote de scoring/simulação de produção, não como holdout de avaliação).
# MAGIC
# MAGIC **Por que Autoloader em vez de `spark.read.csv` direto:**
# MAGIC - Inferência e evolução de schema controladas (`cloudFiles.schemaEvolutionMode`), em vez de
# MAGIC   quebrar silenciosamente se uma coluna nova aparecer num arquivo futuro.
# MAGIC - Checkpoint de progresso — reprocessar o notebook não duplica linhas já ingeridas.
# MAGIC - `trigger(availableNow=True)` processa tudo que está disponível no volume e para — comportamento
# MAGIC   de batch, mas usando a mesma API de streaming que serviria para ingestão contínua depois,
# MAGIC   sem reescrever nada se o volume de dados justificar streaming real no futuro.
# MAGIC
# MAGIC **Convenção Bronze:** schema o mais próximo possível do bruto (sem cast de tipo, sem lógica de
# MAGIC negócio) + colunas de metadata de ingestão. Qualquer decisão de limpeza é responsabilidade do
# MAGIC notebook Silver — Bronze é auditável e reprocessável a qualquer momento.

# COMMAND ----------

dbutils.widgets.text("catalog", "db_risco-credito-databricks", "Catálogo")
dbutils.widgets.text("landing_volume", "landing_volume", "Volume de landing")

catalog = dbutils.widgets.get("catalog")
landing_volume = dbutils.widgets.get("landing_volume")
landing_path = f"/Volumes/{catalog}/bronze/{landing_volume}/credito"

print(f"catalog      = {catalog}")
print(f"landing_path = {landing_path}")

# COMMAND ----------

from pyspark.sql import functions as F

def ingest_bronze(source_file: str, table_name: str, checkpoint_suffix: str):
    """
    Ingestão Autoloader de um CSV específico do volume de landing para uma tabela Bronze.
    Idempotente: reprocessar não duplica (o Autoloader controla isso via checkpoint).
    """
    checkpoint_path = f"/Volumes/{catalog}/bronze/{landing_volume}/_checkpoints/{checkpoint_suffix}"
    schema_location = f"/Volumes/{catalog}/bronze/{landing_volume}/_schemas/{checkpoint_suffix}"

    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", schema_location)
        .option("cloudFiles.schemaEvolutionMode", "rescue")   # colunas novas caem em _rescued_data, não quebram o job
        .option("header", "true")
        .option("pathGlobFilter", source_file)
        .load(landing_path)
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
        .withColumnRenamed("_c0", "source_row_id")   # primeira coluna do CSV do Kaggle é um índice sem nome
    )

    query = (
        df.writeStream.format("delta")
        .option("checkpointLocation", checkpoint_path)
        .outputMode("append")
        .trigger(availableNow=True)
        .toTable(f"`{catalog}`.`bronze`.`{table_name}`")
    )
    query.awaitTermination()
    return spark.table(f"`{catalog}`.`bronze`.`{table_name}`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze — treino (rotulado)

# COMMAND ----------

df_bronze_treino = ingest_bronze(
    source_file="cs-training.csv",
    table_name="credito_treino_raw",
    checkpoint_suffix="credito_treino",
)

spark.sql(f"""
    COMMENT ON TABLE `{catalog}`.`bronze`.`credito_treino_raw` IS
    'Ingestão bruta do Give Me Some Credit (Kaggle) — cs-training.csv, rotulado, via Autoloader'
""")

display(df_bronze_treino.limit(5))
print(f"Linhas ingeridas (treino): {df_bronze_treino.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze — scoring (sem rótulo)
# MAGIC
# MAGIC `cs-test.csv` é o arquivo de submissão original do Kaggle — `SeriousDlqin2yrs` vem vazio.
# MAGIC Não serve como holdout de avaliação (não há como calcular AUC sem rótulo). Papel dele aqui:
# MAGIC simular um lote de produção passando pelo endpoint/pela inferência em lote mais adiante.

# COMMAND ----------

df_bronze_scoring = ingest_bronze(
    source_file="cs-test.csv",
    table_name="credito_scoring_raw",
    checkpoint_suffix="credito_scoring",
)

spark.sql(f"""
    COMMENT ON TABLE `{catalog}`.`bronze`.`credito_scoring_raw` IS
    'Ingestão bruta do Give Me Some Credit (Kaggle) — cs-test.csv, sem rótulo, usado como lote de scoring/simulação de produção'
""")

display(df_bronze_scoring.limit(5))
print(f"Linhas ingeridas (scoring): {df_bronze_scoring.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checagem rápida de _rescued_data
# MAGIC
# MAGIC Se essa coluna vier não-nula em algum registro, é sinal de schema inesperado no CSV
# MAGIC (linha mal formada, coluna extra) — vale investigar antes de seguir pro Silver.

# COMMAND ----------

for tbl in ["credito_treino_raw", "credito_scoring_raw"]:
    rescued_count = (
        spark.table(f"`{catalog}`.`bronze`.`{tbl}`")
        .where(F.col("_rescued_data").isNotNull())
        .count()
    )
    print(f"{tbl}: {rescued_count} linha(s) com _rescued_data não-nulo")
