# src/features/gold.py
# ==============================================================================
# CAMADA GOLD — STAR SCHEMA PARA TREINO E ANÁLISE
# Dataset: Give Me Some Credit
#
# Constrói duas tabelas a partir da Silver (`{catalog}.silver.give_me_some_credit`
# + `{catalog}.silver.give_me_some_credit_scoring`, unidas em uma única Gold):
#   - dim_customer         → customer_id, age, num_dependents
#   - fct_credit_profile   → customer_id + as 10 variáveis explicativas do
#                            README + total_delinquency_events (derivada) +
#                            o target
#
# TREINO + SCORING NA MESMA GOLD (decisão registrada — ver conversa em
# docs/roteiro-execucao.md, Fase 7): o dataset scoring é o holdout sem
# `SeriousDlqin2yrs`, então `target_dlq_2yrs` fica NULO nessas linhas por
# construção — não por erro de dado. É esse nulo que
# `notebooks/ml/03_inferencia_batch.py` e os dois notebooks de
# `notebooks/monitoracao/` usam pra identificar "o que é lote novo a
# pontuar" dentro de `fct_credit_profile`, em vez de precisar de uma
# segunda tabela Gold. A alternativa (Gold separada por dataset) foi
# descartada por exigir reescrever a query dos três notebooks acima, que já
# assumem tabela única.
#
# RECONCILIAÇÃO DE NOMES (Silver consolidada vs. notebooks de análise já
# escritos — resposta_hipotese_1_2_3_4.ipynb e
# 03_analise_shap_threshold_vies_roi.ipynb):
#   A Silver usa nomes descritivos e completos (ver src/ingestion/silver.py).
#   Os notebooks de modelagem já existentes usam os nomes curtos abaixo,
#   hardcoded em SQL e Python. Em vez de reabrir a Silver ou reescrever os
#   notebooks de análise, a Gold faz o de-para — é justamente o papel dela
#   na medallion architecture: entregar o dado no vocabulário de negócio
#   que quem consome (modelo, EDA) já espera, sem vazar decisão de nome
#   técnico da camada de baixo.
#
#     Silver                              -> Gold
#     target_default_2yrs                 -> target_dlq_2yrs
#     revolving_utilization_unsecured      -> revolving_utilization
#     num_open_credit_lines_and_loans      -> num_open_credit_lines
#     num_real_estate_loans_or_lines       -> num_real_estate_loans
#
# `total_delinquency_events` é uma feature DERIVADA (soma das três colunas
# de atraso) — feature engineering pertence à Gold, não à Silver, que só
# limpa/padroniza.
#
# `is_monthly_income_null` e `has_delinquency_outlier` (flags de qualidade
# da Silver) são propositalmente EXCLUÍDAS da Gold: os notebooks de análise
# tratam qualquer coluna não listada em `cols_to_ignore`/`ID_COLS` como
# feature de modelo — incluir as flags aqui as transformaria em feature sem
# ninguém decidir isso conscientemente.
# ==============================================================================

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

DIM_CUSTOMER_COLUMNS = ["customer_id", "age", "num_dependents"]

# Nome Silver -> nome Gold, para as colunas que só mudam de nome
FCT_RENAME_MAP = {
    "monthly_income": "monthly_income",
    "debt_ratio": "debt_ratio",
    "revolving_utilization_unsecured": "revolving_utilization",
    "num_open_credit_lines_and_loans": "num_open_credit_lines",
    "num_real_estate_loans_or_lines": "num_real_estate_loans",
    "num_times_30_59_days_late": "num_times_30_59_days_late",
    "num_times_60_89_days_late": "num_times_60_89_days_late",
    "num_times_90_days_late": "num_times_90_days_late",
    "target_default_2yrs": "target_dlq_2yrs",
}


def _read_silver_combined(
    spark: SparkSession,
    catalog: str,
    silver_schema: str,
    training_table_name: str,
    scoring_table_name: str,
) -> DataFrame:
    """Lê a Silver de treino e, se existir, une a de scoring — que não tem
    `target_default_2yrs` (o CSV de scoring não traz `SeriousDlqin2yrs`), daí
    entrar com esse valor nulo, no mesmo tipo da coluna de treino, antes do
    UNION. Falha alto (não segue silenciosamente) se qualquer OUTRA coluna
    divergir do esperado — schema divergente inesperado é bug de upstream
    (Silver), não algo pra Gold tentar adivinhar como conciliar."""
    training_table = f"{catalog}.{silver_schema}.{training_table_name}"
    scoring_table = f"{catalog}.{silver_schema}.{scoring_table_name}"

    df_training = spark.table(training_table)

    try:
        df_scoring = spark.table(scoring_table)
    except Exception:
        print(f"[AVISO] {scoring_table} não encontrada — Gold construída só com o dataset de treino "
              f"(sem lote de scoring para inferência/monitoramento).")
        return df_training

    if "target_default_2yrs" not in df_scoring.columns:
        target_dtype = dict(df_training.dtypes)["target_default_2yrs"]
        df_scoring = df_scoring.withColumn("target_default_2yrs", F.lit(None).cast(target_dtype))

    colunas_so_no_scoring = set(df_scoring.columns) - set(df_training.columns)
    colunas_so_no_training = set(df_training.columns) - set(df_scoring.columns)
    if colunas_so_no_scoring or colunas_so_no_training:
        raise RuntimeError(
            f"[FALHA] Schema de {scoring_table} diverge do esperado em relação a {training_table} "
            f"além de target_default_2yrs (já tratado). Só no scoring: {colunas_so_no_scoring or '{}'}. "
            f"Só no treino: {colunas_so_no_training or '{}'}. A suposição de que as duas Silver têm o "
            f"mesmo schema (menos o target) não se confirmou — checar src/ingestion/silver.py antes "
            f"de rodar de novo."
        )

    return df_training.unionByName(df_scoring)


def build_dim_customer(df_silver: DataFrame) -> DataFrame:
    """Dimensão de cliente: atributos demográficos, estáveis por customer_id."""
    return (
        df_silver
        .select(*DIM_CUSTOMER_COLUMNS)
        .dropDuplicates(["customer_id"])
        .withColumn("_gold_processed_at", F.current_timestamp())
    )


def build_fct_credit_profile(df_silver: DataFrame) -> DataFrame:
    """Fato de perfil de crédito: as 10 variáveis explicativas do README +
    a feature derivada `total_delinquency_events` + o target, no vocabulário
    de nome que os notebooks de modelagem/EDA já esperam."""
    df = df_silver
    for silver_col, gold_col in FCT_RENAME_MAP.items():
        df = df.withColumnRenamed(silver_col, gold_col)

    return (
        df
        .withColumn(
            "total_delinquency_events",
            F.col("num_times_30_59_days_late")
            + F.col("num_times_60_89_days_late")
            + F.col("num_times_90_days_late"),
        )
        .select(
            "customer_id",
            "monthly_income",
            "debt_ratio",
            "revolving_utilization",
            "num_open_credit_lines",
            "num_real_estate_loans",
            "num_times_30_59_days_late",
            "num_times_60_89_days_late",
            "num_times_90_days_late",
            "total_delinquency_events",
            "target_dlq_2yrs",
        )
        .withColumn("_gold_processed_at", F.current_timestamp())
    )


def run_gold_ingestion(
    spark: SparkSession,
    catalog: str,
    schema: str = "gold",
    silver_schema: str = "silver",
    silver_table_name: str = "give_me_some_credit",
    scoring_table_name: str = "give_me_some_credit_scoring",
    dim_table_name: str = "dim_customer",
    fct_table_name: str = "fct_credit_profile",
) -> None:
    """Orquestra a construção da Gold. Ponto único de entrada para o
    notebook `notebooks/ingestao/03_gold.py`.

    Une a Silver de treino com a de scoring (se existir) antes de construir
    dim/fct — é o que dá origem às linhas com `target_dlq_2yrs` nulo que a
    Fase 7 (inferência batch + monitoramento) espera encontrar.

    Gold aqui é um recálculo determinístico em batch a partir de um
    snapshot completo da Silver — não streaming: a Silver já garante
    incrementalidade/upsert, e a Gold é uma projeção/agregação derivada
    dela, então `overwrite` a cada execução é o padrão correto (evita
    complexidade de merge desnecessária numa camada que não tem estado
    próprio, só reflete a Silver).
    """
    dim_table = f"{catalog}.{schema}.{dim_table_name}"
    fct_table = f"{catalog}.{schema}.{fct_table_name}"

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    df_silver = _read_silver_combined(spark, catalog, silver_schema, silver_table_name, scoring_table_name)

    df_dim = build_dim_customer(df_silver)
    df_fct = build_fct_credit_profile(df_silver)

    (
        df_dim.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(dim_table)
    )
    (
        df_fct.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(fct_table)
    )

    dim_count = spark.table(dim_table).count()
    fct_count = spark.table(fct_table).count()
    target_nulls = spark.table(fct_table).filter(F.col("target_dlq_2yrs").isNull()).count()

    if dim_count == 0 or fct_count == 0:
        raise RuntimeError(
            f"[FALHA] Gold vazia após a ingestão — dim_customer={dim_count:,}, "
            f"fct_credit_profile={fct_count:,}. Confirme se {silver_table} tem dados."
        )

    print(f"[OK] {dim_table} — {dim_count:,} linhas")
    print(f"[OK] {fct_table} — {fct_count:,} linhas ({target_nulls:,} com target nulo)")
