-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 03c — Feature tables finais com PRIMARY KEY nativa
-- MAGIC
-- MAGIC **Case:** `risco-credito-databricks`
-- MAGIC
-- MAGIC *Library notebook* SQL da **mesma pipeline** dos notebooks `02` e `03` (Python) — Lakeflow
-- MAGIC Declarative Pipelines mistura notebooks Python e SQL na mesma pipeline sem problema, cada um
-- MAGIC contribuindo nós ao mesmo grafo de dependências.
-- MAGIC
-- MAGIC **Por que isto é SQL e não Python:** `PRIMARY KEY` só é aceita como parte de uma instrução
-- MAGIC `CREATE MATERIALIZED VIEW`/`CREATE TABLE` — não existe parâmetro equivalente no decorator Python
-- MAGIC `@dlt.table`, e `ALTER TABLE` avulso depois, fora da pipeline, é bloqueado pela Databricks em
-- MAGIC tabelas geridas por Lakeflow Declarative Pipelines. Esta é a forma documentada e suportada de
-- MAGIC declarar a PK: dentro da criação da tabela, não depois.
-- MAGIC
-- MAGIC `${catalog}` é o parâmetro de pipeline (mesmo mecanismo usado nos notebooks Python via
-- MAGIC `spark.conf.get`) — precisa de crase (`` ` ``) ao redor porque o nome do catálogo tem hífen
-- MAGIC (`db_risco-credito-databricks`).

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW `${catalog}`.gold.credito_features_treino (
  source_row_id INT NOT NULL,
  revolving_utilization DOUBLE,
  age INT,
  times_30_59_days_late INT,
  debt_ratio DOUBLE,
  monthly_income DOUBLE,
  open_credit_lines INT,
  times_90_days_late INT,
  real_estate_lines INT,
  times_60_89_days_late INT,
  dependents INT,
  is_outlier_utilization BOOLEAN,
  monthly_income_imputed BOOLEAN,
  dependents_imputed BOOLEAN,
  total_times_late INT,
  has_dependents INT,
  income_per_dependent DOUBLE,
  log_monthly_income DOUBLE,
  credit_lines_per_decade_of_age DOUBLE,
  CONSTRAINT pk_credito_features_treino PRIMARY KEY (source_row_id)
)
COMMENT 'Features derivadas do case Risco de Crédito — base de treino, chave source_row_id, pronta para FeatureLookup'
AS SELECT * FROM `${catalog}`.gold.credito_features_treino_calc;

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW `${catalog}`.gold.credito_features_scoring (
  source_row_id INT NOT NULL,
  revolving_utilization DOUBLE,
  age INT,
  times_30_59_days_late INT,
  debt_ratio DOUBLE,
  monthly_income DOUBLE,
  open_credit_lines INT,
  times_90_days_late INT,
  real_estate_lines INT,
  times_60_89_days_late INT,
  dependents INT,
  is_outlier_utilization BOOLEAN,
  monthly_income_imputed BOOLEAN,
  dependents_imputed BOOLEAN,
  total_times_late INT,
  has_dependents INT,
  income_per_dependent DOUBLE,
  log_monthly_income DOUBLE,
  credit_lines_per_decade_of_age DOUBLE,
  CONSTRAINT pk_credito_features_scoring PRIMARY KEY (source_row_id)
)
COMMENT 'Features derivadas do case Risco de Crédito — lote de scoring (sem rótulo), chave source_row_id, pronta para FeatureLookup'
AS SELECT * FROM `${catalog}`.gold.credito_features_scoring_calc;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Nota de validação:** a lista de colunas acima precisa ficar em sincronia com o que
-- MAGIC `construir_features()` devolve no notebook `03`. Se uma feature nova for adicionada lá, ela
-- MAGIC precisa ser adicionada aqui também — a pipeline falha alto e claro (erro de schema) se as duas
-- MAGIC listas divergirem, em vez de silenciosamente descartar a coluna nova, o que é o comportamento
-- MAGIC desejado para uma tabela com contrato de schema explícito como esta.
