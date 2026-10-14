from mlflow import MlflowClient
c = MlflowClient(registry_uri="databricks-uc")
mv = c.get_model_version_by_alias("credito_hml.gold.credito_risco_xgb", "champion")
print(mv.version, dict(mv.tags))
