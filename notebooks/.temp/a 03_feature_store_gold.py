# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Feature Engineering, Split e Gold (Lakeflow Declarative Pipelines)
# MAGIC
# MAGIC **Case:** `risco-credito-databricks`
# MAGIC
# MAGIC **Mudança de arquitetura:** `construir_features` virou uma função pura reaproveitada dentro de
# MAGIC tabelas `@dlt.table` — mesma lógica de antes, agora materializada pelo Lakeflow. Este notebook é
# MAGIC *library notebook* da **mesma pipeline** do `02_transformacao_silver.py`: por isso as leituras
# MAGIC daqui usam `dlt.read()` (dependência interna ao grafo da pipeline, com lineage automático),
# MAGIC diferente do `02`, que lia a Bronze via `spark.table()` por ela estar fora da pipeline.
# MAGIC
# MAGIC **Correção em relação à primeira versão:** cheguei a propor aplicar `PRIMARY KEY` via
# MAGIC `ALTER TABLE` numa tarefa separada, depois da pipeline rodar. Isso está errado — a documentação
# MAGIC da Databricks é explícita que DDL avulso (`ALTER TABLE`, `COMMENT ON TABLE`) fora da definição da
# MAGIC pipeline é bloqueado em tabelas geridas por Lakeflow Declarative Pipelines
# MAGIC (`STREAMING_TABLE_OPERATION_NOT_ALLOWED.INVALID_ALTER`). `PRIMARY KEY` só é aceita como parte de
# MAGIC uma instrução `CREATE MATERIALIZED VIEW`/`CREATE TABLE` — não existe parâmetro equivalente no
# MAGIC decorator Python `@dlt.table`. Por isso este notebook produz só o **cálculo** das features
# MAGIC (`credito_features_treino_calc`, `credito_features_scoring_calc`, sem PK) e o notebook SQL
# MAGIC `03c_gold_constraints.sql` — *library notebook* desta mesma pipeline — materializa a versão final
# MAGIC com `CONSTRAINT ... PRIMARY KEY` já na criação, habilitando `FeatureLookup` sem precisar do
# MAGIC wrapper legado do Feature Store Client.
# MAGIC
# MAGIC **Split treino/teste — trocado de aleatório com seed para hash determinístico:** a versão
# MAGIC anterior usava `F.rand(seed)` dentro de uma `Window`. Numa pipeline declarativa que pode fazer
# MAGIC *full refresh*, isso é arriscado — se o plano de execução mudar entre execuções, a amostra pode
# MAGIC mudar mesmo com a mesma seed. Troquei para `xxhash64(source_row_id) % 100`, que é
# MAGIC determinístico por linha (não depende de plano de execução nem de ordem): a mesma linha sempre
# MAGIC cai no mesmo lado do split, execução após execução.

# COMMAND ----------

import dlt
from pyspark.sql import functions as F

catalog = spark.conf.get("catalog", "db_risco-credito-databricks")
test_size = float(spark.conf.get("test_size", "0.2"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engenharia de features (função pura, reaproveitada)

# COMMAND ----------

def construir_features(df_silver):
    return df_silver.select(
        "source_row_id",
        "revolving_utilization",
        "age",
        "times_30_59_days_late",
        "debt_ratio",
        "monthly_income",
        "open_credit_lines",
        "times_90_days_late",
        "real_estate_lines",
        "times_60_89_days_late",
        "dependents",
        "is_outlier_utilization",
        "monthly_income_imputed",
        "dependents_imputed",
    ).withColumn(
        "total_times_late",
        F.col("times_30_59_days_late") + F.col("times_60_89_days_late") + F.col("times_90_days_late"),
    ).withColumn(
        "has_dependents", (F.col("dependents") > 0).cast("int")
    ).withColumn(
        "income_per_dependent", F.col("monthly_income") / (F.col("dependents") + F.lit(1))
    ).withColumn(
        "log_monthly_income", F.log1p(F.greatest(F.col("monthly_income"), F.lit(0.0)))
    ).withColumn(
        "credit_lines_per_decade_of_age", F.col("open_credit_lines") / (F.col("age") / F.lit(10.0))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature table — base de treino (cálculo, sem PK ainda)
# MAGIC
# MAGIC `dlt.read()` porque `credito_treino` foi definida no notebook `02`, dentro desta mesma pipeline.
# MAGIC
# MAGIC **Por que o nome termina em `_calc` e não é a feature table final:** a `PRIMARY KEY` só pode ser
# MAGIC declarada dentro de uma instrução `CREATE MATERIALIZED VIEW` (SQL) — não existe parâmetro para
# MAGIC isso no decorator Python `@dlt.table`, e aplicar `ALTER TABLE` depois, fora da pipeline, é
# MAGIC exatamente o padrão que a Databricks orienta a evitar em tabelas geridas por Lakeflow Declarative
# MAGIC Pipelines. Esta tabela aqui é só o cálculo; o notebook `03c_gold_constraints.sql`, mais adiante
# MAGIC na mesma pipeline, materializa a versão final com a PK a partir dela.

# COMMAND ----------

@dlt.table(
    name=f"{catalog}.gold.credito_features_treino_calc",
    comment="Cálculo intermediário — features de treino sem PK; ver credito_features_treino (SQL) para a versão final",
    table_properties={"quality": "gold"},
)
def credito_features_treino_calc():
    df_silver = dlt.read(f"{catalog}.silver.credito_treino")
    return construir_features(df_silver)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split treino/teste — rótulos com hash determinístico estratificado por classe

# COMMAND ----------

@dlt.table(
    name=f"{catalog}.gold.credito_split_labels",
    comment="source_row_id + target + split (train/test), hash determinístico estratificado por classe",
)
def credito_split_labels():
    df_alvo = dlt.read(f"{catalog}.silver.credito_treino").select("source_row_id", "target")

    # Balde 0-99 determinístico por linha — mesmo resultado em toda execução/full refresh
    df_alvo = df_alvo.withColumn("_balde", F.pmod(F.xxhash64("source_row_id"), F.lit(100)))

    corte_test = int(test_size * 100)  # ex.: test_size=0.2 -> baldes 0-19 viram teste, por classe

    return df_alvo.withColumn(
        "split",
        F.when(F.col("_balde") < F.lit(corte_test), "test").otherwise("train"),
    ).drop("_balde")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold — treino e teste
# MAGIC
# MAGIC Duas tabelas separadas (não uma coluna `split` só) porque é assim que o job de treino e a
# MAGIC avaliação de holdout vão consumir — evita todo consumidor downstream ter que lembrar de
# MAGIC filtrar `WHERE split = 'train'`.

# COMMAND ----------

@dlt.table(
    name=f"{catalog}.gold.credito_train",
    comment=f"Split de treino (~{int((1 - test_size) * 100)}%), hash determinístico estratificado por target",
    table_properties={"quality": "gold"},
)
def credito_train():
    df_features = dlt.read(f"{catalog}.gold.credito_features_treino_calc")
    df_split = dlt.read(f"{catalog}.gold.credito_split_labels").where(F.col("split") == "train")
    return df_features.join(df_split.select("source_row_id", "target"), "source_row_id")

# COMMAND ----------

@dlt.table(
    name=f"{catalog}.gold.credito_test",
    comment=f"Split de teste/holdout (~{int(test_size * 100)}%), hash determinístico estratificado por target",
    table_properties={"quality": "gold"},
)
def credito_test():
    df_features = dlt.read(f"{catalog}.gold.credito_features_treino_calc")
    df_split = dlt.read(f"{catalog}.gold.credito_split_labels").where(F.col("split") == "test")
    return df_features.join(df_split.select("source_row_id", "target"), "source_row_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature table — lote de scoring (sem rótulo, cálculo, sem PK ainda)

# COMMAND ----------

@dlt.table(
    name=f"{catalog}.gold.credito_features_scoring_calc",
    comment="Cálculo intermediário — features de scoring sem PK; ver credito_features_scoring (SQL) para a versão final",
    table_properties={"quality": "gold"},
)
def credito_features_scoring_calc():
    df_silver = dlt.read(f"{catalog}.silver.credito_scoring")
    return construir_features(df_silver)
