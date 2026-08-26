# Databricks notebook source
# risco-credito-databricks — Retreino do modelo
import mlflow

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

# TODO: carregar dado tratado (Silver), treinar XGBoost + Regressao Logistica,
#       comparar, logar no MLflow, registrar nova versao no Model Registry
