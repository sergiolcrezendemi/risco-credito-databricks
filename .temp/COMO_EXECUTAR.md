# Como executar o projeto — Risco de Crédito (Databricks)

Passo a passo para rodar o pipeline completo, do ambiente zerado até o
monitoramento em produção. Repositório: `risco-credito-databricks`.

## Visão geral do fluxo

```
Fase 0  Reset do ambiente (opcional)
Fase 1  Bronze — ingestão via Auto Loader
Fase 2  Silver — limpeza e padronização
Fase 3  Gold — star schema
Fase 4  EDA — diagnóstico de qualidade (opcional)
Fase 5  Hipóteses (H1-H4) e perguntas (Q1-Q4) + treino do modelo
Fase 6  Teste de sanidade do modelo registrado
Fase 7  Inferência batch + monitoramento (data drift e concept drift)
```

---

## Pré-requisitos

1. **Acesso ao workspace Databricks** com Unity Catalog habilitado, nos catálogos `credito_dev` (desenvolvimento), `credito_hml` e `credito_prd` (promoções). O host dos três ambientes está em `databricks.yml`.
2. **Repositório clonado** no Databricks (Repos) ou sincronizado localmente com a CLI.
3. **Sincronizar dependências.** Sempre que algo mudar em `pyproject.toml`, rode manualmente numa célula solta, antes de qualquer notebook:
   ```
   %uv sync
   ```
   Isso não está automatizado de propósito — foi uma decisão para manter o fluxo simples.
4. **Convenção de parâmetro.** `catalog` é sempre o primeiro widget de cada notebook: use `credito_dev` em desenvolvimento e `credito_hml`/`credito_prd` nas promoções.

---

## Fase 0 — Reset do ambiente (opcional)

Só rode se for a primeira vez com a estrutura nova ou se as tabelas/checkpoints estiverem contaminados. **Não faz parte do fluxo recorrente.**

| Notebook | Parâmetros |
|---|---|
| `notebooks/ingestao/00_reset_ambiente.py` | `catalog`, `confirmar=RESETAR` |

Apaga (se existirem) `bronze.give_me_some_credit_raw` e `silver.give_me_some_credit` legados, além dos checkpoints antigos. Não toca nos `.csv` em `raw_landing/`.

---

## Fase 1 — Bronze (ingestão via Auto Loader)

Rode o notebook **duas vezes**, uma para cada dataset — eles viram tabelas Bronze separadas de propósito, nunca misturadas:

| Execução | Notebook | Parâmetros | Checar antes de seguir |
|---|---|---|---|
| 1ª | `notebooks/ingestao/01_bronze.py` | `catalog`, `dataset=training` | `spark.table(f"{catalog}.bronze.give_me_some_credit_training_raw").count()` ≈ 150.000 |
| 2ª | `notebooks/ingestao/01_bronze.py` | `catalog`, `dataset=scoring` | Mesma tabela com sufixo `_scoring_raw`, ≈ 101.503 |

---

## Fase 2 — Silver (limpeza e padronização)

| Notebook | Parâmetros | Checar antes de seguir |
|---|---|---|
| `notebooks/ingestao/02_silver.py` | `catalog`, `dataset=training` | `spark.table(f"{catalog}.silver.give_me_some_credit").count()` > 0 |

Se a tabela vier vazia com `[OK]` no log, apague o checkpoint em `/Volumes/{catalog}/silver/checkpoints/give_me_some_credit_training/` e rode de novo.

---

## Fase 3 — Gold (star schema)

| Notebook | Parâmetros | Checar antes de seguir |
|---|---|---|
| `notebooks/ingestao/03_gold.py` | `catalog` | `gold.dim_customer` e `gold.fct_credit_profile` populadas, com a mesma contagem da Silver |

---

## Fase 4 — EDA (opcional, mas recomendado)

Não bloqueia a Fase 5, mas ajuda a pegar problema de dado antes de treinar o modelo em cima dele. Rode `>>>Executar Tudo` em cada notebook:

| Notebook | Parâmetros | O que olhar |
|---|---|---|
| `notebooks/ead/01_eda_bronze.py` | `catalog`, `dataset` | Completude, duplicidade por `customer_id`, cardinalidade |
| `notebooks/ead/02_eda_silver.py` | `catalog` | Distribuição do target, flags de qualidade (`is_monthly_income_null`, `has_delinquency_outlier`) |

---

## Fase 5 — Hipóteses, perguntas e treino do modelo

| Notebook | Parâmetros | Responde | Persiste em |
|---|---|---|---|
| `notebooks/ml/01_hipoteses_h1_h4.py` | `catalog` | H1, H2, H3, H4 | `gold.gold_hypotheses_validation` |
| `notebooks/ml/02_vies_threshold_roi.py` | `catalog` | Q1, Q2, Q3, Q4 | MLflow — registra `{catalog}.gold.credito_risco_xgb`, alias `@champion` |

`02_vies_threshold_roi.py` é o **único** notebook de treino oficial — é ele quem registra o `@champion`.

Checagens específicas:
- ROC-AUC do XGBoost deve ficar por volta de 0,85–0,87.
- H3 (piso de aprovação) e Q2 (custo assimétrico) dão **thresholds diferentes** de propósito — não é bug (ver `src/config/business_params.py`, a fonte única de verdade dos parâmetros financeiros).
- Rodar `02_vies_threshold_roi.py` uma segunda vez deve **carregar** o modelo do Registry, não treinar de novo.

---

## Fase 6 — Teste de sanidade do modelo (antes de confiar no `@champion`)

Não é retreino nem inferência em produção — é a checagem de que o modelo recém-promovido a `@champion` está íntegro.

| O quê | Como | O que esperar |
|---|---|---|
| Carregar pelo alias | `mlflow.xgboost.load_model(f"models:/{catalog}.gold.credito_risco_xgb@champion")` | Carrega sem erro |
| Rodar contra o dataset `scoring` (holdout, sem target) | `modelo.predict(X_scoring)` | Vetor de probabilidades entre 0 e 1, sem `NaN` |
| Conferir se a métrica bate com o Registry | Comparar `auc_roc` do run registrado com o recalculado agora em `X_test` | Mesma ordem de grandeza (~0,85–0,87) — divergência grande indica *training-serving skew* |
| Teste de contrato de schema | Chamar `predict()` removendo uma coluna da `signature` registrada | Deve **falhar** com erro de schema |

---

## Fase 7 — Inferência batch e monitoramento

**Ordem importa:** rode `03_inferencia_batch.py` primeiro — ele gera o lote pontuado que `01_drift_dados.py` usa como "lote atual". `02_concept_drift.py` é independente e pode rodar em simulação a qualquer momento.

| Notebook | Parâmetros | O que faz | Persiste em |
|---|---|---|---|
| `notebooks/ml/03_inferencia_batch.py` | `catalog`, `threshold_aprovacao` (default 0,45) | Aplica o `@champion` sobre os registros novos (Gold com alvo nulo) e decide aprovado/negado | `gold.credito_score_predictions` |
| `notebooks/monitoracao/01_drift_dados.py` | `catalog` | Data drift — PSI e KS por feature, treino vs. lote atual | `ml.monitoramento_drift_dados` + MLflow |
| `notebooks/monitoracao/02_concept_drift.py` | `catalog`, `simulation_mode` (default `true`) | Concept drift — queda de AUC-ROC do `@champion`. Roda em simulação até existir uma tabela de resultados realizados (o rótulo real tem horizonte de 2 anos) | `ml.monitoramento_concept_drift` + MLflow |

Os três notebooks assumem que scoring e treino se distinguem no Gold por `target_dlq_2yrs IS NULL` (scoring) vs. `IS NOT NULL` (treino) — ajustar o `WHERE`/`FROM` nos três se o Silver/Gold real usar tabelas separadas.

---

## Rodar os testes automatizados

A lógica de cada notebook que não depende de Spark/dbutils/MLflow foi extraída para `src/` e é coberta por pytest, sem precisar de cluster:

```
pytest tests/ -v
```

---

## Execução automatizada via Job (Databricks Bundle)

As fases 1, 3, 5 e 7 já estão encadeadas em `resources/risco_credito_pipeline.yml` (10 tasks: `bronze_training`, `bronze_scoring`, `silver`, `silver_scoring`, `gold`, `hipoteses_h1_h4`, `vies_threshold_roi`, `inferencia_batch`, `drift_dados`, `concept_drift`). A Fase 0 (reset) fica fora de propósito, por ser destrutiva e manual; a Fase 6 (teste de sanidade) também não entra, por ainda não ter notebook dedicado.

Escolha **uma** das duas formas de cadastrar o job — nunca as duas, para não duplicar:

**Via Bundle (CLI, recomendado — fica versionado no Git):**
```
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run risco_credito_pipeline -t dev
```
Troque `-t dev` por `-t hml` ou `-t prd` para as outras promoções.

**Via UI (Jobs & Pipelines > Create Job):** replique as 10 tasks e o ambiente manualmente — mais rápido para testar uma vez, mas não fica versionado.

---

## Antes de rodar em produção

- Confirmar a task `silver_scoring` no YAML do job — sem ela, `bronze_scoring` fica órfão e a Fase 7 não encontra dado novo para pontuar.
- Garantir `scipy>=1.10` no ambiente do job (usado pelos dois notebooks de monitoramento).
