# src/quality/profiling.py
# ==============================================================================
# PROFILING REUTILIZÁVEL PARA EDA (Bronze / Silver / Gold)
#
# O diagnóstico original (EDA_bronze.ipynb) tinha essa lógica escrita direto
# no notebook. Isso significa reescrever schema/nulos/duplicidade/cardinalidade
# toda vez que se quer perfilar uma tabela nova (Silver, Gold). Este módulo
# concentra essas checagens uma única vez — os notebooks em notebooks/ead/
# só chamam `run_profile`.
# ==============================================================================

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def null_completeness_report(df: DataFrame) -> DataFrame:
    """Quantidade e percentual de nulos por coluna, ordenado do pior para o melhor."""
    total = df.count()
    exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
    counts = df.agg(*exprs).collect()[0].asDict()

    rows = [
        (col_name, dict(df.dtypes)[col_name], missing, round((missing / total) * 100, 2) if total else 0.0)
        for col_name, missing in counts.items()
    ]
    return (
        df.sparkSession.createDataFrame(rows, ["coluna", "tipo_dado", "qtd_nulos_ou_vazios", "pct_nulos"])
        .orderBy(F.desc("qtd_nulos_ou_vazios"))
    )


def duplicate_report(df: DataFrame, key_cols: list) -> None:
    """Imprime duplicidade exata de linha e duplicidade por chave de negócio."""
    total = df.count()
    exact_dupes = total - df.dropDuplicates().count()
    print(f"Linhas 100% idênticas duplicadas: {exact_dupes:,} ({exact_dupes / total:.2%})" if total else "Tabela vazia.")

    if key_cols:
        key_counts = df.groupBy(*key_cols).count().filter(F.col("count") > 1)
        n_key_dupes = key_counts.count()
        print(f"Contagem de chaves {key_cols} duplicadas: {n_key_dupes:,}")
        if n_key_dupes > 0:
            print(f"Exemplo de ocorrências duplicadas em {key_cols}:")
            key_counts.orderBy(*key_cols).show(5, truncate=False)


def cardinality_report(df: DataFrame) -> DataFrame:
    """Valores distintos aproximados por coluna, para orientar quais colunas
    são categóricas/chave vs. contínuas."""
    total = df.count()
    exprs = [F.approx_count_distinct(F.col(c)).alias(c) for c in df.columns]
    distincts = df.agg(*exprs).collect()[0].asDict()

    rows = [
        (col_name, distinct_count, round((distinct_count / total) * 100, 2) if total else 0.0)
        for col_name, distinct_count in distincts.items()
    ]
    return (
        df.sparkSession.createDataFrame(rows, ["coluna", "valores_distintos_aprox", "pct_distintos"])
        .orderBy("valores_distintos_aprox")
    )


def numeric_summary(df: DataFrame, numeric_cols: list) -> DataFrame:
    """Estatísticas descritivas (min/percentis/max) das colunas numéricas informadas."""
    cols_presentes = [c for c in numeric_cols if c in df.columns]
    return df.select(cols_presentes).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")


def run_profile(
    df: DataFrame,
    table_label: str,
    key_cols: list = None,
    numeric_cols: list = None,
) -> DataFrame:
    """Roda o diagnóstico completo e imprime no padrão do EDA_bronze original.
    Retorna o relatório de nulos (usado para o gráfico de completude no notebook).
    """
    total = df.count()
    print("=" * 60)
    print(f"DIAGNÓSTICO — TABELA: {table_label}")
    print(f"Total de Registros (Linhas): {total:,}")
    print(f"Total de Atributos (Colunas): {len(df.columns)}")
    print("=" * 60)

    print("\n--- SCHEMA ---")
    df.printSchema()

    print("\n--- AUDITORIA DE COMPLETUDE E NULIDADE ---")
    df_nulos = null_completeness_report(df)
    df_nulos.show(truncate=False)

    if key_cols:
        print("\n--- AUDITORIA DE DUPLICIDADE ---")
        duplicate_report(df, key_cols)

    print("\n--- CARDINALIDADE POR COLUNA ---")
    cardinality_report(df).show(truncate=False)

    if numeric_cols:
        print("\n--- SUMÁRIO DE DISTRIBUIÇÃO NUMÉRICA ---")
        numeric_summary(df, numeric_cols).show(truncate=False)

    print("\n--- AMOSTRA DOS DADOS (TOP 5) ---")
    df.limit(5).show(truncate=False)

    return df_nulos
