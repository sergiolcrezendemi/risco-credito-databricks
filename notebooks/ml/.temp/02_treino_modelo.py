# Databricks notebook source
# IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
%uv sync

# COMMAND ----------

# risco-credito-databricks — Retreino do modelo
import mlflow

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

# TODO: carregar dado tratado (Silver), treinar XGBoost + Regressao Logistica,
#       comparar, logar no MLflow, registrar nova versao no Model Registry