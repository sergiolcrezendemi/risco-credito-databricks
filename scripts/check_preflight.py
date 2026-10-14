"""Pré-voo (somente leitura) do catálogo de destino.

Falha ANTES do deploy se a estrutura do catálogo quebraria o pipeline:
  - catálogo, schemas ou volume ausentes;
  - tabela que o pipeline grava como Delta existindo como view, materialized view
    ou streaming table (sobra de arquitetura antiga) — o MERGE falharia no meio da run.

Tabelas que ainda não existem NÃO são erro (o pipeline as cria).

Variáveis de ambiente:
  CATALOG          ex.: credito_prd (default: credito_prd)
  DATABRICKS_HOST / DATABRICKS_TOKEN
"""

import os
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

SCHEMAS = ["bronze", "silver", "gold", "ml"]
VOLUME = "bronze.raw_landing"

# Tabelas que o pipeline grava como Delta (ajuste se o pipeline mudar).
TABELAS_DELTA = {
    "bronze": [
        "give_me_some_credit_training_raw",
        "give_me_some_credit_scoring_raw",
    ],
    "silver": [
        "give_me_some_credit",
        "give_me_some_credit_scoring",
    ],
    "gold": [
        "dim_customer",
        "fct_credit_profile",
        "credito_score_predictions",
    ],
    "ml": [
        "monitoramento_drift_dados",
        "monitoramento_concept_drift",
    ],
}


def _valor(x) -> str:
    """Enums do SDK expõem .value; strings passam direto."""
    return str(getattr(x, "value", x))


def conflitos(tabelas_por_schema: dict) -> list:
    """Devolve as tabelas esperadas como Delta que existem com outro tipo/formato.

    tabelas_por_schema: {schema: [(nome, tipo, formato), ...]}
    """
    problemas = []
    for schema, esperadas in TABELAS_DELTA.items():
        existentes = {n: (t, f) for n, t, f in tabelas_por_schema.get(schema, [])}
        for nome in esperadas:
            if nome in existentes:
                tipo, formato = existentes[nome]
                if tipo != "MANAGED" or formato != "DELTA":
                    problemas.append(
                        f"{schema}.{nome}: tipo={tipo}, formato={formato} (esperado MANAGED/DELTA)"
                    )
    return problemas


def avisos(tabelas_por_schema: dict) -> list:
    """Objetos fora do padrão que o pipeline não usa (não bloqueiam)."""
    out = []
    for schema, tabelas in tabelas_por_schema.items():
        esperadas = set(TABELAS_DELTA.get(schema, []))
        for nome, tipo, _ in tabelas:
            if tipo != "MANAGED" and nome not in esperadas:
                out.append(f"{schema}.{nome} ({tipo})")
    return out


def main() -> int:
    catalogo = os.environ.get("CATALOG", "credito_prd")
    w = WorkspaceClient()
    problemas = []

    try:
        w.catalogs.get(catalogo)
    except NotFound:
        print(f"[PRÉ-VOO] REPROVADO: catálogo '{catalogo}' não existe.")
        return 1

    tabelas_por_schema = {}
    for schema in SCHEMAS:
        try:
            w.schemas.get(f"{catalogo}.{schema}")
        except NotFound:
            problemas.append(f"schema {catalogo}.{schema} não existe")
            continue
        tabelas_por_schema[schema] = [
            (t.name, _valor(t.table_type), _valor(t.data_source_format))
            for t in w.tables.list(catalog_name=catalogo, schema_name=schema)
        ]

    try:
        w.volumes.read(f"{catalogo}.{VOLUME}")
    except NotFound:
        problemas.append(f"volume {catalogo}.{VOLUME} não existe")

    problemas += conflitos(tabelas_por_schema)

    for item in avisos(tabelas_por_schema):
        print(f"[PRÉ-VOO] aviso (não bloqueia): objeto fora do padrão {item}")

    if problemas:
        print(f"[PRÉ-VOO] REPROVADO em {catalogo}:")
        for p in problemas:
            print(f"  - {p}")
        return 1

    print(f"[PRÉ-VOO] APROVADO: estrutura de {catalogo} compatível com o pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
