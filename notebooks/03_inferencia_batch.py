# Databricks notebook source
# risco-credito-databricks — Job de inferencia em lote
import mlflow
dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")
# TODO: model = mlflow.pyfunc.load_model(f"models:/{catalog}.gold.modelo/Production")
