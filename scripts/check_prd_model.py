"""Smoke test pós-deploy: confirma que o ambiente tem o modelo com o alias esperado.

Em PRD também exige as tags de linhagem da promoção (source_version).
Em HML (onde o modelo é treinado, não promovido) use REQUIRE_LINEAGE=false.

Variáveis de ambiente:
  MODEL_FULL_NAME  ex.: credito_prd.gold.credito_risco_xgb
  ALIAS            (opcional) default: champion
  REQUIRE_LINEAGE  (opcional) default: true — exige a tag source_version
  DATABRICKS_HOST / DATABRICKS_TOKEN
"""

import os
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def main() -> int:
    name = os.environ["MODEL_FULL_NAME"]
    alias = os.environ.get("ALIAS", "champion")
    exigir_linhagem = os.environ.get("REQUIRE_LINEAGE", "true").lower() == "true"
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")

    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException as e:
        print(f"FALHA: alias '{alias}' não encontrado (ou sem acesso/credencial) em {name}: {e}")
        return 1

    tags = mv.tags or {}
    if exigir_linhagem and not tags.get("source_version"):
        print(f"FALHA: {name} v{mv.version} sem tag de linhagem (source_version).")
        return 1

    if not tags.get("algoritmo"):
        print(f"FALHA: {name} v{mv.version} sem tag 'algoritmo' (a inferência depende dela).")
        return 1

    origem = (
        f"origem {tags.get('source_model')} v{tags['source_version']}, " if exigir_linhagem else ""
    )
    print(
        f"OK: {name}@{alias} = v{mv.version} "
        f"({origem}algoritmo={tags['algoritmo']}, "
        f"auc_roc={tags.get('auc_roc', tags.get('auc_roc_hml', 'n/d'))})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
