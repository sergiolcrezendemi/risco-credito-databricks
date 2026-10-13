"""Gate de qualidade de HML.

Falha (exit 1) se o modelo recém-treinado (@challenger, sempre a última versão
registrada pelo treino) não existir ou tiver métrica abaixo do mínimo.
Lê as métricas gravadas como tags da versão no Unity Catalog pelo notebook
04_treino_mlflow.py. Credenciais: DATABRICKS_HOST / DATABRICKS_TOKEN.
"""

import os
import sys

import mlflow

MODEL_FULL_NAME = os.environ.get("MODEL_FULL_NAME", "credito_hml.gold.credito_risco_xgb")
METRIC_NAME = os.environ.get("METRIC_NAME", "auc_roc")
MIN_METRIC = float(os.environ.get("MIN_METRIC", "0.80"))
ALIAS = os.environ.get("MODEL_ALIAS", "challenger")


def main() -> int:
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")

    try:
        mv = client.get_model_version_by_alias(MODEL_FULL_NAME, ALIAS)
    except Exception as e:
        print(f"[GATE] REPROVADO: {MODEL_FULL_NAME}@{ALIAS} não encontrado. O treino rodou? Erro: {e}")
        return 1

    bruto = mv.tags.get(METRIC_NAME)
    if bruto is None:
        print(f"[GATE] REPROVADO: versão {mv.version} sem a tag '{METRIC_NAME}'. Tags: {dict(mv.tags)}")
        return 1

    valor = float(bruto)
    algoritmo = mv.tags.get("algoritmo", "?")
    print(f"[GATE] {MODEL_FULL_NAME} v{mv.version} ({algoritmo}) | {METRIC_NAME}={valor:.4f} | mínimo={MIN_METRIC:.2f}")

    if valor < MIN_METRIC:
        print("[GATE] REPROVADO: métrica abaixo do mínimo.")
        return 1

    print("[GATE] APROVADO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
