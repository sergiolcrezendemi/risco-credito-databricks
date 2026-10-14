-- =============================================================================
-- 00_infra_hml.sql
-- Infraestrutura de HML (idempotente). Rodar ANTES do primeiro deploy/pipeline,
-- no SQL Editor do Databricks. O pipeline assume que estes objetos existem.
-- Para PRD, copiar trocando credito_hml por credito_prd.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS credito_hml.bronze;
CREATE SCHEMA IF NOT EXISTS credito_hml.silver;
CREATE SCHEMA IF NOT EXISTS credito_hml.gold;
CREATE SCHEMA IF NOT EXISTS credito_hml.ml;

CREATE VOLUME IF NOT EXISTS credito_hml.bronze.raw_landing;
CREATE VOLUME IF NOT EXISTS credito_hml.bronze.checkpoints;
CREATE VOLUME IF NOT EXISTS credito_hml.bronze.schemas;
CREATE VOLUME IF NOT EXISTS credito_hml.silver.checkpoints;

-- Conferência
SHOW SCHEMAS IN credito_hml;
