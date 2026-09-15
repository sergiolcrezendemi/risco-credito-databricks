# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# # risco-credito-databricks — Ingestao Bronze
# dbutils.widgets.text("catalog", "credito_dev")
# catalog = dbutils.widgets.get("catalog")
# # TODO: apontar para a fonte real (ver link da base no README.md)

# COMMAND ----------

# IMPORTANTE
#  SEMPRE DEFINIR OS TIPOS DOS CAMPOS DO ARQUIVO QUE SERÁ INSERIDO - MELHOR PRÁTICA DE MERCADO


# Databricks notebook source
# ==============================================================================
# INGESTÃO INCREMENTAL COM AUTO LOADER (CLOUD FILES) - FREE EDITION SAFE
# Fornecer o schema explicitamente elimina o consumo excessivo de memória
# ==============================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, LongType
)

CATALOG = "credito_prd"
SCHEMA = "bronze"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.give_me_some_credit_raw"

RAW_DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing/"
CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/give_me_some_credit_bronze/"
SCHEMA_LOCATION = f"/Volumes/{CATALOG}/{SCHEMA}/schemas/give_me_some_credit_bronze/"

# 1. Definir o schema explícito (evita inferência em streaming)
# cs-training possui 12 colunas (incluindo o índice inicial _c0)
credit_schema = StructType([
    StructField("_c0", IntegerType(), True),
    StructField("SeriousDlqin2yrs", IntegerType(), True),
    StructField("RevolvingUtilizationOfUnsecuredLines", DoubleType(), True),
    StructField("age", IntegerType(), True),
    StructField("NumberOfTime30-59DaysPastDueNotWorse", IntegerType(), True),
    StructField("DebtRatio", DoubleType(), True),
    StructField("MonthlyIncome", DoubleType(), True),
    StructField("NumberOfOpenCreditLinesAndLoans", IntegerType(), True),
    StructField("NumberOfTimes90DaysLate", IntegerType(), True),
    StructField("NumberRealEstateLoansOrLines", IntegerType(), True),
    StructField("NumberOfTime60-89DaysPastDueNotWorse", IntegerType(), True),
    StructField("NumberOfDependents", DoubleType(), True)
])

# 2. Leitura com cloudFiles sem inferSchema
df_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", "true")
    .schema(credit_schema) # Schema fornecido: zero custo de inferência
    .option("cloudFiles.schemaLocation", SCHEMA_LOCATION)
    .option("pathGlobFilter", "*.csv") # Ignora o .xls
    .load(RAW_DATA_PATH)
    .withColumnRenamed("_c0", "customer_id")
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_name"))
)

# 3. Escrita incremental com trigger availableNow (modo micro-batch)
query = (
    df_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(availableNow=True)
    .toTable(TABLE_NAME)
)

query.awaitTermination()

print(f"Ingestão via Auto Loader concluída com sucesso na tabela {TABLE_NAME}!")

# COMMAND ----------

# # Databricks notebook source
# # ==============================================================================
# # PIPELINE DE INGESTÃO INCREMENTAL COM DEDUPLICAÇÃO (DELTA MERGE)
# # Dataset: Give Me Some Credit
# # ==============================================================================

# from pyspark.sql import functions as F
# from pyspark.sql.window import Window
# from delta.tables import DeltaTable

# # 1. Configurações de Governança no Unity Catalog
# CATALOG = "credito_prd"
# SCHEMA = "bronze"
# TABLE_NAME = f"{CATALOG}.{SCHEMA}.give_me_some_credit_raw"
# RAW_DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing/"

# # 2. Leitura dos dados brutos com metadados de arquivo
# df_incoming = (
#     spark.read
#     .format("csv")
#     .option("header", "true")
#     .option("inferSchema", "true")
#     .option("pathGlobFilter", "*.csv")
#     .load(RAW_DATA_PATH)
#     .withColumnRenamed("_c0", "customer_id") # Padroniza a primeira coluna como ID único
#     .withColumn("_ingestion_timestamp", F.current_timestamp())
#     .withColumn("_source_file", F.col("_metadata.file_name"))
#     .withColumn("_file_modification_time", F.col("_metadata.file_modification_time"))
# )

# # 3. Deduplicação do lote de entrada (mantém o registro mais recente caso haja duplicidade no arquivo)
# window_spec = Window.partitionBy("customer_id").orderBy(F.col("_file_modification_time").desc())
# df_deduplicated = (
#     df_incoming
#     .withColumn("row_num", F.row_number().over(window_spec))
#     .filter(F.col("row_num") == 1)
#     .drop("row_num")
# )

# # 4. Criação da tabela Delta na primeira execução (caso não exista)
# if not spark.catalog.tableExists(TABLE_NAME):
#     (
#         df_deduplicated.write
#         .format("delta")
#         .mode("overwrite")
#         .saveAsTable(TABLE_NAME)
#     )
#     print(f"Tabela {TABLE_NAME} criada pela primeira vez com {df_deduplicated.count()} registros.")
# else:
#     # 5. Execução do Delta MERGE (Upsert) - Evita duplicações e atualiza modificações
#     target_table = DeltaTable.forName(spark, TABLE_NAME)
    
#     (
#         target_table.alias("target")
#         .merge(
#             source=df_deduplicated.alias("source"),
#             condition="target.customer_id = source.customer_id"
#         )
#         .whenMatchedUpdateAll() # Atualiza os campos se o registro já existir
#         .whenNotMatchedInsertAll() # Insere se for um novo cliente
#         .execute()
#     )
#     print(f"Merge incremental executado com sucesso na tabela {TABLE_NAME}.")

# # 6. Validação de integridade: checagem de unicidade por customer_id
# total_registros = spark.table(TABLE_NAME).count()
# total_unicos = spark.table(TABLE_NAME).select("customer_id").distinct().count()
# print(f"Registros totais: {total_registros} | IDs únicos: {total_unicos}")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# # Databricks notebook source
# # ==============================================================================
# # PIPELINE DE INGESTÃO - CAMADA BRONZE (AUTO LOADER)
# # Dataset: Give Me Some Credit
# # ==============================================================================

# from pyspark.sql import functions as F

# # 1. Configurações de Governança no Unity Catalog
# CATALOG = "credito_prd"
# SCHEMA = "bronze"
# TABLE_NAME = f"{CATALOG}.{SCHEMA}.give_me_some_credit_raw"

# # 2. Criação e Validação dos Contêineres de Dados e Volumes
# spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
# spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_landing")
# spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints")
# spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.schemas")

# # 3. Definição dos Caminhos no Unity Catalog Volumes
# # Certifique-se de que os arquivos foram colocados neste caminho (ou na raiz do volume)
# RAW_DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing/"
# CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/give_me_some_credit_bronze/"
# SCHEMA_LOCATION = f"/Volumes/{CATALOG}/{SCHEMA}/schemas/give_me_some_credit_bronze/"

# # 4. Leitura Streaming com Auto Loader
# df_raw_stream = (
#     spark.readStream
#     .format("cloudFiles")
#     .option("cloudFiles.format", "csv")
#     .option("header", "true")
#     .option("inferSchema", "true")
#     .option("cloudFiles.schemaLocation", SCHEMA_LOCATION)
#     .option("pathGlobFilter", "*.csv")
#     .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
#     .load(RAW_DATA_PATH)
# )




# # 5. Enriquecimento com Metadados de Auditoria
# df_bronze = (
#     df_raw_stream
#     .withColumn("_ingestion_timestamp", F.current_timestamp())
#     .withColumn("_source_file", F.col("_metadata.file_name"))
#     .withColumn("_file_modification_time", F.col("_metadata.file_modification_time"))
# )

# # 6. Escrita Incremental na Tabela Delta Lake
# query = (
#     df_bronze.writeStream
#     .format("delta")
#     .outputMode("append")
#     .option("checkpointLocation", CHECKPOINT_PATH)
#     .option("mergeSchema", "true")
#     .trigger(availableNow=True)
#     .toTable(TABLE_NAME)
# )




# query.awaitTermination()

# print(f"Ingestão concluída com sucesso na tabela: {TABLE_NAME}")

# COMMAND ----------



# COMMAND ----------

# # Databricks notebook source
# # ==============================================================================
# # PIPELINE DE INGESTÃO - CAMADA BRONZE (AUTO LOADER)
# # Dataset: Give Me Some Credit (Kaggle)
# # Formato: Streaming incremental com trigger once / availableNow
# # ==============================================================================

# from pyspark.sql import functions as F

# # 1. Configurações de Governança no Unity Catalog e Paths de Armazenamento
# CATALOG = "credit_prod"
# SCHEMA = "bronze"
# TABLE_NAME = f"{CATALOG}.{SCHEMA}.give_me_some_credit_raw"

# # Caminhos no Cloud Storage / DBFS / Volume do Unity Catalog
# RAW_DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing/give_me_some_credit/"
# CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/give_me_some_credit_bronze/"
# SCHEMA_LOCATION = f"/Volumes/{CATALOG}/{SCHEMA}/schemas/give_me_some_credit_bronze/"

# # 2. Garantir a existência do Catálogo e Schema
# spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# # 3. Leitura Streaming com Databricks Auto Loader
# df_raw_stream = (
#     spark.readStream
#     .format("cloudFiles")
#     .option("cloudFiles.format", "csv")
#     .option("header", "true")
#     .option("inferSchema", "true")
#     .option("cloudFiles.schemaLocation", SCHEMA_LOCATION)
#     .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
#     .load(RAW_DATA_PATH)
# )

# # 4. Enriquecimento com Metadados de Auditoria da Camada Bronze
# # Na camada Bronze, os dados brutos são mantidos intactos, adicionando rastreabilidade
# df_bronze = (
#     df_raw_stream
#     .withColumn("_ingestion_timestamp", F.current_timestamp())
#     .withColumn("_source_file", F.col("_metadata.file_name"))
#     .withColumn("_file_modification_time", F.col("_metadata.file_modification_time"))
# )

# # 5. Escrita Incremental na Tabela Delta Lake com Trigger Once/AvailableNow
# query = (
#     df_bronze.writeStream
#     .format("delta")
#     .outputMode("append")
#     .option("checkpointLocation", CHECKPOINT_PATH)
#     .option("mergeSchema", "true")
#     .trigger(availableNow=True)
#     .toTable(TABLE_NAME)
# )

# query.awaitTermination()

# print(f"Ingestão concluída com sucesso na tabela: {TABLE_NAME}")