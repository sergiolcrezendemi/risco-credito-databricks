# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup de Catálogo, Schemas e Volume (Unity Catalog)
# MAGIC
# MAGIC **Case:** `risco-credito-databricks`
# MAGIC **Objetivo:** criar a estrutura de governança (catálogo → schemas bronze/silver/gold → volume de landing
# MAGIC dentro de `bronze`) antes de qualquer ingestão.
# MAGIC
# MAGIC Estrutura alvo:
# MAGIC ```
# MAGIC db_risco-credito-databricks
# MAGIC ├── bronze
# MAGIC │   └── landing_volume   (Volume — CSVs brutos pousam aqui)
# MAGIC ├── silver
# MAGIC └── gold
# MAGIC ```
# MAGIC
# MAGIC **Nota sobre o nome do catálogo:** Unity Catalog aceita hífen no nome, mas exige que toda
# MAGIC referência a ele em SQL venha entre crases (`` `db_risco-credito-databricks` ``) — é por isso que
# MAGIC todo o script usa f-string com backticks em vez de referência solta. Os widgets abaixo continuam
# MAGIC parametrizáveis (para reuso em outro ambiente/nome sem editar código), só os valores padrão
# MAGIC é que já vêm preenchidos com a estrutura combinada.

# COMMAND ----------

dbutils.widgets.text("catalog", "db_risco-credito-databricks", "Catálogo")
dbutils.widgets.text("landing_volume", "landing_volume", "Nome do volume de landing (raw CSV)")

catalog = dbutils.widgets.get("catalog")
landing_volume = dbutils.widgets.get("landing_volume")

print(f"catalog       = {catalog}")
print(f"landing_volume = {landing_volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catálogo e schemas
# MAGIC
# MAGIC `IF NOT EXISTS` em tudo — o notebook é idempotente, pode rodar de novo sem quebrar
# MAGIC (importante porque ele entra no job de deploy do Asset Bundle, não só é rodado manualmente uma vez).

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}` COMMENT 'Case Risco de Crédito — Databricks de Ponta a Ponta'")

for schema, comment in [
    ("bronze", "Dado bruto ingerido via Autoloader, schema tal como recebido + metadados de ingestão"),
    ("silver", "Dado limpo, tipado e com regras de qualidade aplicadas"),
    ("gold", "Feature table para treino/inferência e conjuntos de treino/teste/scoring"),
    ("models", "Model Registry (Unity Catalog) — versões registradas, aliases @champion/@challenger"),
]:
    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`
        COMMENT '{comment}'
    """)

print("Catálogo e schemas bronze/silver/gold/models prontos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume de landing (raw)
# MAGIC
# MAGIC É para onde os CSVs brutos (`cs-training.csv`, `cs-test.csv`, `Data_Dictionary.xls`) devem ser
# MAGIC copiados **antes** de rodar o notebook `01_ingestao_bronze_autoloader`. O Autoloader lê a partir
# MAGIC daqui — não do Unity Catalog Volumes de outro schema, e não de `/dbfs` (deprecado para dado novo).
# MAGIC
# MAGIC Upload local → volume, via Databricks CLI (fora do notebook):
# MAGIC ```bash
# MAGIC databricks fs cp cs-training.csv dbfs:/Volumes/{catalog}/bronze/{landing_volume}/credito/cs-training.csv
# MAGIC databricks fs cp cs-test.csv     dbfs:/Volumes/{catalog}/bronze/{landing_volume}/credito/cs-test.csv
# MAGIC ```
# MAGIC Ou arrastar o arquivo direto na UI do Catalog Explorer, dentro do volume criado abaixo.

# COMMAND ----------

spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS `{catalog}`.`bronze`.`{landing_volume}`
    COMMENT 'Landing zone de arquivos brutos (CSV) antes da ingestão via Autoloader'
""")

landing_path = f"/Volumes/{catalog}/bronze/{landing_volume}/credito"
dbutils.fs.mkdirs(landing_path)

print(f"Volume pronto em: {landing_path}")
print("Copie cs-training.csv e cs-test.csv para esse path antes de rodar o notebook 01.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RBAC (placeholder de governança)
# MAGIC
# MAGIC Em produção, o grant real deve restringir escrita em `gold` só ao service principal do job
# MAGIC de treino/inferência, e leitura de `bronze` só a quem faz curadoria. Ajustar os grupos abaixo
# MAGIC para os grupos reais do workspace antes de aplicar em prd.

# COMMAND ----------

# Exemplo — ajustar nomes de grupo reais antes de rodar em hml/prd:
# spark.sql(f"GRANT USE CATALOG ON CATALOG `{catalog}` TO `data-science-team`")
# spark.sql(f"GRANT SELECT ON SCHEMA `{catalog}`.`gold` TO `data-science-team`")
# spark.sql(f"GRANT SELECT ON SCHEMA `{catalog}`.`silver` TO `data-science-team`")
# spark.sql(f"GRANT ALL PRIVILEGES ON SCHEMA `{catalog}`.`bronze` TO `data-eng-team`")

print("RBAC: revisar grants acima e descomentar com os grupos reais do workspace antes de prd.")
