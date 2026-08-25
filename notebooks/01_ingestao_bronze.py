# Databricks notebook source
# risco-credito-databricks — Ingestao Bronze
dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")
# TODO: apontar para a fonte real (ver link da base no README.md)
