"""Smoke test pós-deploy: confirma que PRD tem o modelo promovido com linhagem.

Variáveis de ambiente:
  MODEL_FULL_NAME  ex.: credito_prd.gold.credito_risco_xgb
  ALIAS            (opcional) default: champion
  DATABRICKS_HOST / DATABRICKS_TOKEN
"""

import os
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def main() -> int:
    name = os.environ["MODEL_FULL_NAME"]
    alias = os.environ.get("ALIAS", "champion")
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")

    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException as e:
        print(f"FALHA: alias '{alias}' não encontrado em {name}: {e}")
        return 1

    tags = mv.tags or {}
    if not tags.get("source_version"):
        print(f"FALHA: {name} v{mv.version} sem tag de linhagem (source_version).")
        return 1

    print(
        f"OK: {name}@{alias} = v{mv.version} "
        f"(origem {tags.get('source_model')} v{tags['source_version']}, "
        f"algoritmo={tags.get('algoritmo', 'n/d')}, "
        f"auc_roc_hml={tags.get('auc_roc_hml', 'n/d')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
