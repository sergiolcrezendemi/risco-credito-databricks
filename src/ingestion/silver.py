# src/ingestion/silver.py
# ==============================================================================
# CAMADA SILVER — LIMPEZA, PADRONIZAÇÃO E MERGE INCREMENTAL
# Dataset: Give Me Some Credit
#
# Este módulo é a FONTE ÚNICA DA VERDADE para o schema e as regras de limpeza
# da Silver. Tanto o notebook de ingestão (notebooks/ingestao/02_silver.py)
# quanto o de EDA (notebooks/ead/*) devem importar daqui — nunca reimplementar
# a renomeação de colunas ou as regras de sanidade localmente.
#
# CONSOLIDAÇÃO DE DIVERGÊNCIAS (entre 02__ingestao_silver.ipynb e ead-silver.txt):
#   - Nome do target: adotado `target_default_2yrs` (mais descritivo que
#     `target_dlq_2yrs`).
#   - Faixa de idade válida: adotado 18–120 (mais permissivo; registros fora
#     da faixa são colocados em quarentena lógica, não descartados).
#   - Nomes de colunas: adotada a forma mais descritiva do ead-silver.txt
#     (`revolving_utilization_unsecured`, `num_open_credit_lines_and_loans`,
#     `num_real_estate_loans_or_lines`) em vez das abreviações do notebook.
#   - Estratégia de qualidade: adotada a abordagem de QUARENTENA (flags +
#     `None` no valor inválido) do ead-silver.txt em vez do `.filter()` que
#     descartava linhas silenciosamente no notebook original — preserva o
#     registro para auditoria e não distorce contagens de linhas Bronze→Silver.
# ==============================================================================

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType
from delta.tables import DeltaTable

# ------------------------------------------------------------------------------
# Mapeamento único de colunas (snake_case) — ver rationale de consolidação acima
# ------------------------------------------------------------------------------
COLUMN_MAPPING = {
    "SeriousDlqin2yrs": "target_default_2yrs",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization_unsecured",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "num_times_30_59_days_late",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "num_open_credit_lines_and_loans",
    "NumberOfTimes90DaysLate": "num_times_90_days_late",
    "NumberRealEstateLoansOrLines": "num_real_estate_loans_or_lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "num_times_60_89_days_late",
    "NumberOfDependents": "num_dependents",
}

MIN_AGE = 18
MAX_AGE = 120
DELINQUENCY_OUTLIER_THRESHOLD = 96  # valores >= 96 são códigos de erro conhecidos do dataset


def rename_columns(df: DataFrame) -> DataFrame:
    """Aplica o COLUMN_MAPPING único do projeto."""
    df_renamed = df
    for old_col, new_col in COLUMN_MAPPING.items():
        if old_col in df_renamed.columns:
            df_renamed = df_renamed.withColumnRenamed(old_col, new_col)
    return df_renamed


def clean_silver_schema(df_bronze: DataFrame) -> DataFrame:
    """Padroniza nomes, tipa colunas e aplica regras de qualidade por
    quarentena lógica (sem descarte prematuro de linhas)."""
    df = rename_columns(df_bronze)

    return (
        df
        .filter(F.col("customer_id").isNotNull())
        # Idade fora da faixa plausível vira null (quarentena), não é descartada
        .withColumn(
            "age",
            F.when((F.col("age") < MIN_AGE) | (F.col("age") > MAX_AGE), None)
             .otherwise(F.col("age")),
        )
        .withColumn("is_monthly_income_null", F.when(F.col("monthly_income").isNull(), 1).otherwise(0))
        .withColumn("monthly_income", F.col("monthly_income").cast(DoubleType()))
        .withColumn("num_dependents", F.coalesce(F.col("num_dependents").cast(IntegerType()), F.lit(0)))
        .withColumn(
            "has_delinquency_outlier",
            F.when(
                (F.col("num_times_30_59_days_late") >= DELINQUENCY_OUTLIER_THRESHOLD) |
                (F.col("num_times_60_89_days_late") >= DELINQUENCY_OUTLIER_THRESHOLD) |
                (F.col("num_times_90_days_late") >= DELINQUENCY_OUTLIER_THRESHOLD),
                1,
            ).otherwise(0),
        )
        .withColumn("_silver_processed_at", F.current_timestamp())
    )


def _dedup_batch(batch_df: DataFrame) -> DataFrame:
    """Garante um único registro por customer_id dentro do micro-batch,
    mantendo o mais recente por _ingestion_timestamp."""
    window_spec = Window.partitionBy("customer_id").orderBy(F.col("_ingestion_timestamp").desc())
    return (
        batch_df
        .withColumn("_row_num", F.row_number().over(window_spec))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


def upsert_to_silver(batch_df: DataFrame, batch_id: int, silver_table: str) -> None:
    """Função de micro-batch (foreachBatch) que aplica MERGE idempotente na Silver."""
    batch_deduped = _dedup_batch(batch_df)

    if not batch_deduped.sparkSession.catalog.tableExists(silver_table):
        batch_deduped.write.format("delta").mode("overwrite").saveAsTable(silver_table)
        return

    silver_delta = DeltaTable.forName(batch_deduped.sparkSession, silver_table)
    (
        silver_delta.alias("tgt")
        .merge(batch_deduped.alias("src"), "tgt.customer_id = src.customer_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def run_silver_ingestion(
    spark: SparkSession,
    catalog: str,
    schema: str = "silver",
    bronze_schema: str = "bronze",
    table_name: str = "give_me_some_credit",
) -> None:
    """Orquestra a ingestão Silver completa. Ponto único de entrada para o
    notebook `notebooks/ingestao/02_silver.py`.

    Corrige o bug observado no notebook original: o volume de checkpoint
    agora é garantido ANTES do início do writeStream.
    """
    bronze_table = f"{catalog}.{bronze_schema}.{table_name}_raw"
    silver_table = f"{catalog}.{schema}.{table_name}"
    checkpoint_path = f"/Volumes/{catalog}/{schema}/checkpoints/{table_name}/"

    # 1. Garante schema e volume ANTES de qualquer referência no writeStream
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.checkpoints")

    # 2. Leitura incremental da Bronze
    df_bronze_stream = spark.readStream.format("delta").table(bronze_table)

    # 3. Limpeza e padronização (fonte única — ver clean_silver_schema)
    df_silver_clean = clean_silver_schema(df_bronze_stream)

    # 4. Upsert idempotente via MERGE
    def _batch_fn(batch_df, batch_id):
        upsert_to_silver(batch_df, batch_id, silver_table)

    query_silver = (
        df_silver_clean.writeStream
        .format("delta")
        .foreachBatch(_batch_fn)
        .option("checkpointLocation", checkpoint_path)
        .trigger(availableNow=True)
        .start()
    )
    query_silver.awaitTermination()

    print(f"[OK] Carga Silver concluída em {silver_table}")
