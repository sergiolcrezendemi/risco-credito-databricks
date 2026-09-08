# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Transformação Silver (Lakeflow Declarative Pipelines)
# MAGIC
# MAGIC **Case:** `risco-credito-databricks`
# MAGIC
# MAGIC **Mudança de arquitetura em relação à versão anterior:** este notebook deixou de escrever
# MAGIC `writeStream`/`MERGE` manual e virou uma **Lakeflow Declarative Pipeline** (antigo Delta Live
# MAGIC Tables) — cada tabela é uma função decorada com `@dlt.table`, e o Lakeflow cuida de
# MAGIC orquestração, idempotência e evolução incremental sozinho. É a recomendação atual da Databricks
# MAGIC para pipelines de ingestão/transformação novos.
# MAGIC
# MAGIC **O que o Lakeflow assume no lugar do código manual:**
# MAGIC - Idempotência/reprocessamento — antes fazíamos `MERGE` por `source_row_id` na mão; o Lakeflow
# MAGIC   gerencia isso internamente a cada atualização da pipeline.
# MAGIC - Constraints de qualidade — antes eram `ALTER TABLE ... ADD CONSTRAINT` aplicados depois da
# MAGIC   escrita; agora são `@dlt.expect*` declarados junto da tabela, com métricas visíveis no
# MAGIC   event log da pipeline (dashboard de qualidade nativo).
# MAGIC
# MAGIC **O que continua igual:** a lógica de negócio das decisões de qualidade documentadas antes
# MAGIC (idade zero descartada, utilização > 10 marcada não removida, imputação de renda por mediana
# MAGIC por faixa etária, imputação de dependentes pela moda).
# MAGIC
# MAGIC **Nota operacional:** este notebook é registrado como *library notebook* de uma Lakeflow
# MAGIC Declarative Pipeline (não roda como notebook Job comum). O `01_ingestao_bronze_autoloader.py`
# MAGIC continua fora da pipeline — ingestão Bronze via Autoloader simples orquestrada por Job, como já
# MAGIC estava — por isso a leitura da Bronze aqui usa `spark.table()` (fonte externa à pipeline) em vez
# MAGIC de `dlt.read()` (reservado para dependências internas ao grafo desta mesma pipeline, como o
# MAGIC notebook `03_feature_store_gold.py` vai fazer ao ler as tabelas Silver definidas aqui).

# COMMAND ----------

import dlt
from pyspark.sql import functions as F

# Parâmetro de pipeline (equivalente ao widget dos notebooks Job) — configurado na seção
# "configuration" das settings da pipeline, com fallback para o valor combinado.
catalog = spark.conf.get("catalog", "db_risco-credito-databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Função de transformação (reaproveitada nas duas tabelas Silver)

# COMMAND ----------

def transformar_silver(df_bronze):
    """
    Cast de tipos, imputação documentada e flags de qualidade sobre o dado Bronze.
    Mesma lógica de negócio da versão anterior — só a forma de materializar mudou.
    """
    df = df_bronze.select(
        F.col("source_row_id").cast("int").alias("source_row_id"),
        F.col("SeriousDlqin2yrs").cast("int").alias("target"),
        F.col("RevolvingUtilizationOfUnsecuredLines").cast("double").alias("revolving_utilization"),
        F.col("age").cast("int").alias("age"),
        F.col("`NumberOfTime30-59DaysPastDueNotWorse`").cast("int").alias("times_30_59_days_late"),
        F.col("DebtRatio").cast("double").alias("debt_ratio"),
        F.col("MonthlyIncome").cast("double").alias("monthly_income"),
        F.col("NumberOfOpenCreditLinesAndLoans").cast("int").alias("open_credit_lines"),
        F.col("NumberOfTimes90DaysLate").cast("int").alias("times_90_days_late"),
        F.col("NumberRealEstateLoansOrLines").cast("int").alias("real_estate_lines"),
        F.col("`NumberOfTime60-89DaysPastDueNotWorse`").cast("int").alias("times_60_89_days_late"),
        F.col("NumberOfDependents").cast("int").alias("dependents"),
        F.col("_ingested_at"),
        F.col("_source_file"),
    )

    df = df.withColumn(
        "is_outlier_utilization",
        F.when(F.col("revolving_utilization") > 10, F.lit(True)).otherwise(F.lit(False)),
    )

    faixa_etaria = F.when(F.col("age") < 30, "18-29") \
        .when(F.col("age") < 45, "30-44") \
        .when(F.col("age") < 60, "45-59") \
        .otherwise("60+")
    df = df.withColumn("faixa_etaria_tmp", faixa_etaria)

    mediana_por_faixa = (
        df.where(F.col("monthly_income").isNotNull())
        .groupBy("faixa_etaria_tmp")
        .agg(F.expr("percentile_approx(monthly_income, 0.5)").alias("mediana_renda"))
    )

    df = df.join(mediana_por_faixa, on="faixa_etaria_tmp", how="left")
    df = df.withColumn("monthly_income_imputed", F.col("monthly_income").isNull())
    df = df.withColumn(
        "monthly_income",
        F.when(F.col("monthly_income").isNull(), F.col("mediana_renda")).otherwise(F.col("monthly_income")),
    )
    df = df.drop("faixa_etaria_tmp", "mediana_renda")

    df = df.withColumn("dependents_imputed", F.col("dependents").isNull())
    df = df.withColumn("dependents", F.coalesce(F.col("dependents"), F.lit(0)))

    df = df.withColumn("_silver_processed_at", F.current_timestamp())

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver — treino
# MAGIC
# MAGIC `age <= 0` agora é regra declarativa (`expect_or_drop`) em vez de `.where()` manual — a linha
# MAGIC descartada fica registrada nas métricas de qualidade da pipeline, visível no dashboard, em vez
# MAGIC de desaparecer silenciosamente. `is_outlier_utilization` e a checagem pós-imputação viram
# MAGIC `expect` (monitoram e contam, mas não descartam) — mantém a decisão de negócio já tomada de
# MAGIC não remover esses registros, só sinalizar.

# COMMAND ----------

@dlt.table(
    name=f"{catalog}.silver.credito_treino",
    comment="Dado de treino limpo, tipado e com flags de qualidade — chave source_row_id",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("age_valida", "age IS NULL OR age > 0")
@dlt.expect("utilizacao_dentro_do_esperado", "is_outlier_utilization = false")
@dlt.expect("sem_nulos_pos_imputacao", "monthly_income IS NOT NULL AND dependents IS NOT NULL")
def credito_treino():
    df_bronze = spark.table(f"{catalog}.bronze.credito_treino_raw")
    return transformar_silver(df_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver — scoring
# MAGIC
# MAGIC Mesma função de transformação, aplicada sobre `cs-test.csv` (sem rótulo). A mediana de renda
# MAGIC é recalculada sobre este próprio lote — não vaza estatística do treino para o scoring.

# COMMAND ----------

@dlt.table(
    name=f"{catalog}.silver.credito_scoring",
    comment="Lote de scoring (cs-test.csv, sem rótulo) limpo e tipado — simula lote de produção",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("age_valida", "age IS NULL OR age > 0")
@dlt.expect("utilizacao_dentro_do_esperado", "is_outlier_utilization = false")
@dlt.expect("sem_nulos_pos_imputacao", "monthly_income IS NOT NULL AND dependents IS NOT NULL")
def credito_scoring():
    df_bronze = spark.table(f"{catalog}.bronze.credito_scoring_raw")
    return transformar_silver(df_bronze)
