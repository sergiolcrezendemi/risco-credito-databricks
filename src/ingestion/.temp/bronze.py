# src/ingestion/bronze.py
# ==============================================================================
# CAMADA BRONZE — INGESTÃO INCREMENTAL COM AUTO LOADER (CLOUD FILES)
# Dataset: Give Me Some Credit
#
# Este módulo concentra a lógica de ingestão Bronze para ser importada pelos
# notebooks de orquestração (notebooks/ingestao/01_bronze.py). Os notebooks
# não devem reimplementar este código — apenas chamar `run_bronze_ingestion`.
# ==============================================================================

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType,
)

# ------------------------------------------------------------------------------
# Schema explícito do CSV de origem (cs-training / Give Me Some Credit)
# Fornecer o schema evita o custo de inferência em streaming (best practice
# para Auto Loader em ambientes Free Edition / custo controlado).
# ------------------------------------------------------------------------------
CREDIT_SCHEMA = StructType([
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
    StructField("NumberOfDependents", DoubleType(), True),
])


def ensure_bronze_infra(spark: SparkSession, catalog: str, schema: str = "bronze") -> None:
    """Garante schema e volumes necessários antes de qualquer leitura/escrita.

    Corrige o padrão observado no notebook Silver original, onde o volume de
    checkpoint era referenciado antes de existir (UC_VOLUME_NOT_FOUND).
    """
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_landing")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.checkpoints")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.schemas")


def build_bronze_stream(
    spark: SparkSession,
    raw_data_path: str,
    schema_location: str,
) -> DataFrame:
    """Lê os CSVs de origem via Auto Loader e enriquece com metadados de linhagem."""
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .schema(CREDIT_SCHEMA)
        .option("cloudFiles.schemaLocation", schema_location)
        .option("pathGlobFilter", "*.csv")
        .load(raw_data_path)
        .withColumnRenamed("_c0", "customer_id")
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_name"))
    )


def write_bronze_incremental(
    df_stream: DataFrame,
    target_table: str,
    checkpoint_path: str,
) -> StreamingQuery:
    """Escreve o stream Bronze em modo append, com trigger availableNow
    (micro-batch único, seguro e econômico para Free Edition)."""
    return (
        df_stream.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .trigger(availableNow=True)
        .toTable(target_table)
    )


def run_bronze_ingestion(
    spark: SparkSession,
    catalog: str,
    schema: str = "bronze",
    table_name: str = "give_me_some_credit",
) -> None:
    """Orquestra a ingestão Bronze completa. Ponto único de entrada para o
    notebook `notebooks/ingestao/01_bronze.py`.

    Parameters
    ----------
    catalog: nome do catálogo Unity Catalog (ex.: "credito_dev", "credito_prd")
             — deve vir de um widget no notebook, nunca hardcoded.
    """
    target_table = f"{catalog}.{schema}.{table_name}_raw"
    raw_data_path = f"/Volumes/{catalog}/{schema}/raw_landing/"
    checkpoint_path = f"/Volumes/{catalog}/{schema}/checkpoints/{table_name}_bronze/"
    schema_location = f"/Volumes/{catalog}/{schema}/schemas/{table_name}_bronze/"

    ensure_bronze_infra(spark, catalog, schema)

    df_stream = build_bronze_stream(spark, raw_data_path, schema_location)
    query = write_bronze_incremental(df_stream, target_table, checkpoint_path)
    query.awaitTermination()

    print(f"[OK] Ingestão Bronze concluída em {target_table}")
