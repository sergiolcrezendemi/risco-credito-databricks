from mlflow import MlflowClient

c = MlflowClient(registry_uri="databricks-uc")
mv = c.get_model_version_by_alias("credito_hml.gold.credito_risco_xgb", "champion")

print(f'tags : {mv.version, dict(mv.tags)} \n')
print(f'auc-roc : {mv.version, mv.tags.get("auc_roc")} \n')

