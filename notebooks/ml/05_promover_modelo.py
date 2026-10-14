# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Promoção de modelo (HML → PRD)
# MAGIC
# MAGIC Copia a versão `champion` do modelo aprovado em homologação para o catálogo deste ambiente,
# MAGIC **sem retreinar** (mesmo artefato validado = rastreabilidade/auditoria).
# MAGIC
# MAGIC - Idempotente: se a versão de origem já foi promovida, reaproveita e só garante o alias.
# MAGIC - Registra tags de linhagem (modelo/versão/run de origem, métrica, data).
# MAGIC - Rollback: apontar o alias `champion` de volta para a versão anterior.

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_prd")
dbutils.widgets.text("source_catalog", "credito_hml")
dbutils.widgets.text("schema", "gold")
dbutils.widgets.text("model_name", "credito_risco_xgb")
dbutils.widgets.text("source_alias", "champion")   # alias lido em HML
dbutils.widgets.text("alias", "champion")          # alias gravado em PRD

catalog = dbutils.widgets.get("catalog")
source_catalog = dbutils.widgets.get("source_catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
source_alias = dbutils.widgets.get("source_alias")
alias = dbutils.widgets.get("alias")

if catalog == source_catalog:
    raise ValueError("catalog e source_catalog iguais: promoção deve cruzar ambientes.")

src_name = f"{source_catalog}.{schema}.{model_name}"
dst_name = f"{catalog}.{schema}.{model_name}"
print(f"Origem : {src_name}@{source_alias}")
print(f"Destino: {dst_name}@{alias}")

# COMMAND ----------

from datetime import datetime, timezone

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

client = MlflowClient(registry_uri="databricks-uc")

# 1) Versão aprovada em HML (falha explícita se não houver alias)
try:
    src_mv = client.get_model_version_by_alias(src_name, source_alias)
except MlflowException as e:
    raise RuntimeError(
        f"Alias '{source_alias}' não encontrado em {src_name}. "
        "Rode o pipeline de HML (treino) antes de promover para PRD."
    ) from e

print(f"Versão aprovada em HML: v{src_mv.version} (run_id={src_mv.run_id})")

# COMMAND ----------

# 2) Idempotência: já existe em PRD uma cópia desta versão de origem?
existing = None
try:
    for mv in client.search_model_versions(f"name='{dst_name}'"):
        tags = mv.tags or {}
        if tags.get("source_model") == src_name and tags.get("source_version") == str(src_mv.version):
            existing = mv
            break
except MlflowException:
    pass  # modelo ainda não existe em PRD

# COMMAND ----------

# 3) Copia (se necessário)
if existing is None:
    try:
        client.get_registered_model(dst_name)
    except MlflowException:
        client.create_registered_model(dst_name)

    new_mv = client.copy_model_version(
        src_model_uri=f"models:/{src_name}/{src_mv.version}",
        dst_name=dst_name,
    )

    lineage_tags = {
        "source_model": src_name,
        "source_version": str(src_mv.version),
        "source_run_id": src_mv.run_id or "",
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Métrica validada em HML (se disponível no run de origem)
    try:
        auc = client.get_run(src_mv.run_id).data.metrics.get("auc_roc")
        if auc is not None:
            lineage_tags["auc_roc_hml"] = f"{auc:.4f}"
    except Exception as e:  # noqa: BLE001 - métrica é informativa, não bloqueia
        print(f"Aviso: não foi possível ler auc_roc do run de origem: {e}")

    # Carrega TODAS as tags de HML (ex.: `algoritmo`, que a inferência usa para escolher o
    # flavor xgboost/sklearn, e as métricas) e soma as tags de linhagem da promoção.
    tags_finais = {**(src_mv.tags or {}), **lineage_tags}
    for k, v in tags_finais.items():
        client.set_model_version_tag(dst_name, new_mv.version, k, v)

    target_version = new_mv.version
    print(f"Copiado: {dst_name} v{target_version}")
else:
    target_version = existing.version
    print(f"Versão de origem já promovida: {dst_name} v{target_version} (nada a copiar)")

# COMMAND ----------

# 4) Alias de produção
client.set_registered_model_alias(dst_name, alias, target_version)
print(f"Alias '{alias}' -> {dst_name} v{target_version}")
