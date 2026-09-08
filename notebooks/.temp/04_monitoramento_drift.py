# Databricks notebook source
# risco-credito-databricks — Monitoramento de drift
dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")
# TODO: configurar Lakehouse Monitoring sobre a tabela Gold de scores
