"""Dicionário de dados do projeto risco_credito_databricks — FONTE ÚNICA.

Este módulo é a única fonte de verdade para a descrição das colunas do Silver
e do Gold. Dele saem dois artefatos, sempre sincronizados:

1. Metadados no Unity Catalog (COMMENT de tabela/coluna e tags) —
   `apply_dictionary` / `apply_all`, chamados no fim do pipeline.
2. `docs/dicionario_de_dados.md` — gerado por `render_markdown`
   (`python -m src.config.data_dictionary`). Não editar o .md à mão.

Sem imports de pyspark: roda em pytest sem cluster. O objeto `spark` é só
recebido por parâmetro.

Escopo: Silver e Gold. O Bronze mantém os nomes originais do Kaggle e não é
documentado aqui.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_TAG = "risco_credito_databricks"

# Colunas ocultas de metadado de arquivo do Databricks — não são do schema.
_HIDDEN_COLUMNS = {"_metadata", "_object_metadata"}


@dataclass(frozen=True)
class Column:
    name: str
    description: str
    source: str = ""          # origem/linhagem, ex.: "Kaggle: SeriousDlqin2yrs"
    optional: bool = False    # True = pode não existir na tabela sem ser erro


@dataclass(frozen=True)
class Table:
    name: str                 # sem catálogo/schema
    schema: str               # silver | gold
    description: str
    columns: tuple[Column, ...]
    tags: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Regras de limpeza aplicadas no Silver (documentadas no .md gerado).
# ATENÇÃO: extraídas da versão de 09/09 do silver.py — conferir contra
# src/ingestion/silver.py sempre que a limpeza mudar.
# ------------------------------------------------------------------------------
SILVER_CLEANING_RULES = (
    "Deduplicação por `customer_id`.",
    "`age` fora do intervalo 18–115 é convertida em nulo.",
    "`num_dependents` nulo é convertido em 0.",
    "`is_monthly_income_null` = 1 quando `monthly_income` é nulo (a renda NÃO é imputada).",
    "`has_delinquency_outlier` = 1 quando qualquer coluna de atraso "
    "(30-59, 60-89, 90+) é maior ou igual a 96.",
    "Nenhuma linha é descartada: anomalias viram flag (quarentena lógica).",
)

# ------------------------------------------------------------------------------
# SILVER
# ------------------------------------------------------------------------------
SILVER_COLUMNS: tuple[Column, ...] = (
    Column("customer_id",
           "Identificador do cliente (chave de negócio; a tabela é deduplicada por esta coluna).",
           "Kaggle: customer_id"),
    Column("target_default_2yrs",
           "Alvo: 1 = atraso de 90 dias ou mais nos últimos 2 anos; 0 = sem atraso. "
           "Nulo/ausente no dataset de scoring.",
           "Kaggle: SeriousDlqin2yrs"),
    Column("revolving_utilization_unsecured",
           "Saldo total em cartões de crédito e linhas pessoais (exceto imóveis e "
           "parcelados) dividido pela soma dos limites de crédito.",
           "Kaggle: RevolvingUtilizationOfUnsecuredLines"),
    Column("age",
           "Idade do cliente em anos. Valores fora de 18–115 são anulados.",
           "Kaggle: age"),
    Column("num_times_30_59_days_late",
           "Número de vezes com atraso de 30 a 59 dias nos últimos 2 anos.",
           "Kaggle: NumberOfTime30-59DaysPastDueNotWorse"),
    Column("debt_ratio",
           "Pagamentos mensais de dívidas, pensão e custos de vida divididos pela "
           "renda bruta mensal.",
           "Kaggle: DebtRatio"),
    Column("monthly_income",
           "Renda mensal do cliente. Pode ser nula (ver is_monthly_income_null).",
           "Kaggle: MonthlyIncome"),
    Column("num_open_credit_lines_and_loans",
           "Número de empréstimos abertos (ex.: financiamento de carro) e linhas de "
           "crédito (ex.: cartões).",
           "Kaggle: NumberOfOpenCreditLinesAndLoans"),
    Column("num_times_90_days_late",
           "Número de vezes com atraso de 90 dias ou mais.",
           "Kaggle: NumberOfTimes90DaysLate"),
    Column("num_real_estate_loans_or_lines",
           "Número de financiamentos imobiliários e linhas de crédito com garantia de imóvel.",
           "Kaggle: NumberRealEstateLoansOrLines"),
    Column("num_times_60_89_days_late",
           "Número de vezes com atraso de 60 a 89 dias nos últimos 2 anos.",
           "Kaggle: NumberOfTime60-89DaysPastDueNotWorse"),
    Column("num_dependents",
           "Número de dependentes do cliente (exceto ele próprio). Nulo vira 0 no Silver.",
           "Kaggle: NumberOfDependents"),
    Column("is_monthly_income_null",
           "Flag de qualidade: 1 = renda mensal nula na origem; 0 = informada. "
           "Não é propagada para o Gold.",
           "Derivada (Silver)"),
    Column("has_delinquency_outlier",
           "Flag de qualidade: 1 = alguma coluna de atraso com valor >= 96 (valor atípico "
           "da origem); 0 = sem outlier. Não é propagada para o Gold.",
           "Derivada (Silver)"),
    Column("_ingestion_timestamp",
           "Momento em que a linha foi ingerida no Bronze (linhagem).",
           "Metadado de pipeline"),
    Column("_source_file",
           "Arquivo de origem da linha (linhagem).",
           "Metadado de pipeline"),
    Column("_silver_processed_at",
           "Momento em que a linha foi processada para o Silver (linhagem).",
           "Metadado de pipeline"),
)

_SILVER_BY_NAME = {c.name: c for c in SILVER_COLUMNS}

# No dataset de scoring o alvo pode não existir: não é erro.
_SILVER_SCORING_COLUMNS = tuple(
    replace(c, optional=True) if c.name == "target_default_2yrs" else c
    for c in SILVER_COLUMNS
)


# ------------------------------------------------------------------------------
# GOLD — descrições reaproveitadas do Silver (sem duplicar texto).
# ------------------------------------------------------------------------------
def _from_silver(silver_name: str, gold_name: str | None = None) -> Column:
    """Coluna do Gold herdando a descrição do Silver; só a origem muda."""
    base = _SILVER_BY_NAME[silver_name]
    return Column(
        name=gold_name or silver_name,
        description=base.description,
        source=f"Silver: {silver_name}",
    )


GOLD_DIM_CUSTOMER_COLUMNS: tuple[Column, ...] = (
    _from_silver("customer_id"),
    _from_silver("age"),
    _from_silver("num_dependents"),
    Column("_gold_processed_at",
           "Momento em que a linha foi processada para o Gold (linhagem).",
           "Metadado de pipeline"),
)

GOLD_FCT_CREDIT_PROFILE_COLUMNS: tuple[Column, ...] = (
    _from_silver("customer_id"),
    _from_silver("monthly_income"),
    _from_silver("debt_ratio"),
    _from_silver("revolving_utilization_unsecured", "revolving_utilization"),
    _from_silver("num_open_credit_lines_and_loans", "num_open_credit_lines"),
    _from_silver("num_real_estate_loans_or_lines", "num_real_estate_loans"),
    _from_silver("num_times_30_59_days_late"),
    _from_silver("num_times_60_89_days_late"),
    _from_silver("num_times_90_days_late"),
    Column("total_delinquency_events",
           "Feature derivada: soma dos atrasos de 30-59, 60-89 e 90+ dias.",
           "Derivada (Gold): num_times_30_59_days_late + num_times_60_89_days_late "
           "+ num_times_90_days_late"),
    _from_silver("target_default_2yrs", "target_dlq_2yrs"),
)

# ------------------------------------------------------------------------------
# TABELAS
# ------------------------------------------------------------------------------
TABLES: tuple[Table, ...] = (
    Table(
        name="give_me_some_credit",
        schema="silver",
        description="Silver do dataset Give Me Some Credit (treino): dados limpos, "
                    "padronizados e deduplicados por customer_id, com flags de qualidade.",
        columns=SILVER_COLUMNS,
        tags={"camada": "silver", "projeto": PROJECT_TAG},
    ),
    Table(
        name="give_me_some_credit_scoring",
        schema="silver",
        description="Silver do dataset Give Me Some Credit (scoring): lote sem rótulo "
                    "a ser pontuado pelo modelo. Mesmo schema do treino.",
        columns=_SILVER_SCORING_COLUMNS,
        tags={"camada": "silver", "projeto": PROJECT_TAG},
    ),
    Table(
        name="dim_customer",
        schema="gold",
        description="Dimensão de cliente: atributos demográficos estáveis por customer_id.",
        columns=GOLD_DIM_CUSTOMER_COLUMNS,
        tags={"camada": "gold", "projeto": PROJECT_TAG},
    ),
    Table(
        name="fct_credit_profile",
        schema="gold",
        description="Fato de perfil de crédito: variáveis explicativas do modelo, "
                    "feature derivada total_delinquency_events e o alvo. Inclui as linhas "
                    "de scoring (alvo nulo) quando o Silver de scoring existe.",
        columns=GOLD_FCT_CREDIT_PROFILE_COLUMNS,
        tags={"camada": "gold", "projeto": PROJECT_TAG},
    ),
)


# ------------------------------------------------------------------------------
# UNITY CATALOG
# ------------------------------------------------------------------------------
def _q(text: str) -> str:
    """Literal string do Spark SQL (escape com barra invertida)."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def apply_dictionary(spark, catalog: str, table: Table, strict: bool = False) -> dict:
    """Aplica COMMENT de tabela/colunas e tags no Unity Catalog.

    Reaplicar a cada execução do pipeline: um CREATE OR REPLACE / overwrite
    pode descartar os comentários.

    Retorna um relatório com:
      - `undocumented`: colunas da tabela que não estão no dicionário;
      - `missing_in_table`: colunas do dicionário (não opcionais) ausentes na tabela.
    Com `strict=True`, qualquer divergência levanta RuntimeError (depois de aplicar
    os comentários possíveis).
    """
    fqn = f"{catalog}.{table.schema}.{table.name}"

    if not spark.catalog.tableExists(fqn):
        print(f"[AVISO] {fqn} não existe — dicionário não aplicado.")
        return {"table": fqn, "skipped": True, "undocumented": [], "missing_in_table": []}

    existing = [c for c in spark.table(fqn).columns if c not in _HIDDEN_COLUMNS]
    documented = {c.name for c in table.columns}

    undocumented = [c for c in existing if c not in documented]
    missing = [c.name for c in table.columns if c.name not in existing and not c.optional]

    spark.sql(f"COMMENT ON TABLE {fqn} IS {_q(table.description)}")
    for col in table.columns:
        if col.name in existing:
            spark.sql(f"ALTER TABLE {fqn} ALTER COLUMN `{col.name}` COMMENT {_q(col.description)}")

    if table.tags:
        tags_sql = ", ".join(f"{_q(k)} = {_q(v)}" for k, v in table.tags.items())
        try:
            spark.sql(f"ALTER TABLE {fqn} SET TAGS ({tags_sql})")
        except Exception as exc:  # tag exige APPLY TAG; comentário é o essencial
            print(f"[AVISO] Tags não aplicadas em {fqn}: {exc}")

    if undocumented:
        print(f"[AVISO] {fqn}: colunas sem descrição no dicionário: {undocumented}")
    if missing:
        print(f"[AVISO] {fqn}: colunas do dicionário ausentes na tabela: {missing}")
    if strict and (undocumented or missing):
        raise RuntimeError(
            f"[FALHA] Dicionário divergente de {fqn}. Sem descrição: {undocumented}. "
            f"Ausentes na tabela: {missing}."
        )

    print(f"[OK] Dicionário aplicado em {fqn}")
    return {"table": fqn, "skipped": False, "undocumented": undocumented,
            "missing_in_table": missing}


def apply_all(spark, catalog: str, schemas: tuple[str, ...] = ("silver", "gold"),
              strict: bool = False) -> list[dict]:
    """Aplica o dicionário em todas as tabelas das camadas indicadas."""
    return [apply_dictionary(spark, catalog, t, strict=strict)
            for t in TABLES if t.schema in schemas]


# ------------------------------------------------------------------------------
# MARKDOWN
# ------------------------------------------------------------------------------
def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown() -> str:
    lines = [
        "# Dicionário de dados — risco_credito_databricks",
        "",
        "> Gerado automaticamente a partir de `src/config/data_dictionary.py`. "
        "**Não editar à mão** — altere o módulo e rode "
        "`python -m src.config.data_dictionary`.",
        "> Os mesmos textos são aplicados como `COMMENT` no Unity Catalog "
        "(`apply_all`), que é a fonte consultável no Catalog Explorer.",
        "",
        "## Regras de limpeza (Silver)",
        "",
    ]
    lines += [f"- {rule}" for rule in SILVER_CLEANING_RULES]

    for t in TABLES:
        lines += ["", f"## `<catalog>.{t.schema}.{t.name}`", "", t.description, ""]
        if t.tags:
            lines += ["Tags: " + ", ".join(f"`{k}={v}`" for k, v in t.tags.items()), ""]
        lines += ["| Coluna | Descrição | Origem |", "|---|---|---|"]
        for c in t.columns:
            lines.append(f"| `{c.name}` | {_cell(c.description)} | {_cell(c.source)} |")

    return "\n".join(lines) + "\n"


def _docs_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "dicionario_de_dados.md"


def main(argv: list[str]) -> int:
    """Sem argumentos: escreve o .md. Com `--check`: falha se o .md estiver desatualizado."""
    target = _docs_path()
    content = render_markdown()
    if "--check" in argv:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            print(f"[FALHA] {target} desatualizado. Rode: python -m src.config.data_dictionary")
            return 1
        print("[OK] docs/dicionario_de_dados.md está em dia.")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"[OK] {target} gerado.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
