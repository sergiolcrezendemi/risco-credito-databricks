# src/ingestion/maintenance.py
# ==============================================================================
# UTILITÁRIOS DE MANUTENÇÃO — não fazem parte do pipeline de produção.
# Usado uma única vez para limpar o estado gerado pela estrutura antiga
# (raw_landing sem subpastas training/scoring, misturando cs-training.csv
# e cs-test.csv na mesma tabela Bronze).
# ==============================================================================

from pyspark.sql import SparkSession

# Nomes gerados pela estrutura ANTIGA (antes de separar training/scoring).
# Se você rodar isso num catálogo que nunca usou a estrutura antiga, tudo
# aqui será um no-op seguro (todas as checagens usam "if exists").
LEGACY_BRONZE_TABLE = "give_me_some_credit_raw"
LEGACY_SILVER_TABLE = "give_me_some_credit"
LEGACY_BRONZE_CHECKPOINT = "give_me_some_credit_bronze"
LEGACY_BRONZE_SCHEMA_LOC = "give_me_some_credit_bronze"
LEGACY_SILVER_CHECKPOINT = "give_me_some_credit"


def _rm_if_exists(dbutils, path: str) -> str:
    """Remove um diretório em Volumes se ele existir. Retorna uma mensagem
    de status (nunca lança erro se o caminho não existir)."""
    try:
        dbutils.fs.ls(path)  # dispara FileNotFoundException se não existir
    except Exception:
        return f"  (nada a remover) {path}"
    dbutils.fs.rm(path, recurse=True)
    return f"  [removido] {path}"


def reset_legacy_environment(
    spark: SparkSession,
    dbutils,
    catalog: str,
    confirm: bool,
    bronze_schema: str = "bronze",
    silver_schema: str = "silver",
) -> None:
    """Apaga tabelas e checkpoints da estrutura ANTIGA (pré subpastas
    training/scoring), para permitir reingestão limpa com o novo layout.

    Parameters
    ----------
    confirm: precisa ser True explicitamente. É a trava de segurança contra
             execução acidental — este utilitário faz DROP TABLE e rm
             recursivo, ações destrutivas e irreversíveis.
    """
    if not confirm:
        raise ValueError(
            "reset_legacy_environment não executou nada. Passe confirm=True "
            "explicitamente para confirmar que você quer apagar as tabelas "
            "e checkpoints legados listados no docstring deste módulo."
        )

    print(f"[RESET] Ambiente: catalog={catalog}")

    # 1. Tabelas legadas (Unity Catalog)
    bronze_table = f"{catalog}.{bronze_schema}.{LEGACY_BRONZE_TABLE}"
    silver_table = f"{catalog}.{silver_schema}.{LEGACY_SILVER_TABLE}"

    spark.sql(f"DROP TABLE IF EXISTS {bronze_table}")
    print(f"  [removido se existia] {bronze_table}")

    spark.sql(f"DROP TABLE IF EXISTS {silver_table}")
    print(f"  [removido se existia] {silver_table}")

    # 2. Checkpoints e schemaLocation legados (Volumes)
    print(_rm_if_exists(
        dbutils, f"/Volumes/{catalog}/{bronze_schema}/checkpoints/{LEGACY_BRONZE_CHECKPOINT}/"
    ))
    print(_rm_if_exists(
        dbutils, f"/Volumes/{catalog}/{bronze_schema}/schemas/{LEGACY_BRONZE_SCHEMA_LOC}/"
    ))
    print(_rm_if_exists(
        dbutils, f"/Volumes/{catalog}/{silver_schema}/checkpoints/{LEGACY_SILVER_CHECKPOINT}/"
    ))

    print(
        "[OK] Reset concluído. Agora rode 01_bronze.py com dataset=training "
        "e dataset=scoring, depois 02_silver.py."
    )
