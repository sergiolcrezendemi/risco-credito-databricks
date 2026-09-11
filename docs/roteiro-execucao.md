# Roteiro de Execução — Risco de Crédito

Ordem de execução do projeto, do ambiente zerado até a validação das
hipóteses (H1-H4) e perguntas (Q1-Q4) do `README.md`. Lista viva — cresce
conforme novas fases (inferência batch, monitoramento) ficam prontas.

Convenção: `catalog` é sempre o primeiro widget de cada notebook —
`credito_dev` em desenvolvimento, `credito_hml`/`credito_prd` nas
promoções.

---

## Pré-requisito (uma vez, ou quando mudar dependência)

Se você adicionou algo novo ao `pyproject.toml` (raiz do repo), sincronize
o ambiente antes de rodar qualquer notebook:

```
%uv sync
```

Digite manualmente numa célula solta quando precisar — não faz parte do
código dos notebooks (ver decisão registrada: mantivemos simples, sem
detecção automática).

---

## Fase 0 — Reset do ambiente (opcional, só quando necessário)

| Notebook | Parâmetros | Quando rodar |
|---|---|---|
| `notebooks/ingestao/00_reset_ambiente.py` | `catalog`, `confirmar=RESETAR` | Só na primeira vez com a estrutura nova, ou se precisar zerar tabelas/checkpoints contaminados. **Não** faz parte do fluxo recorrente. |

Apaga (se existirem) `bronze.give_me_some_credit_raw` e
`silver.give_me_some_credit` legados + checkpoints antigos. Não toca nos
`.csv` em `raw_landing/`.

---

## Fase 1 — Bronze (ingestão via Autoloader)

| Notebook | Parâmetros | Checar antes de seguir |
|---|---|---|
| `notebooks/ingestao/01_bronze.py` | `catalog`, `dataset=training` | `spark.table(f"{catalog}.bronze.give_me_some_credit_training_raw").count()` ≈ 150.000 |
| `notebooks/ingestao/01_bronze.py` (de novo) | `catalog`, `dataset=scoring` | Mesma tabela, sufixo `_scoring_raw`, ≈ 101.503 |

Rode os **dois** — `training` (com target) e `scoring` (holdout, sem
target) viram tabelas Bronze separadas de propósito, nunca misturadas.

---

## Fase 2 — Silver (limpeza e padronização)

| Notebook | Parâmetros | Checar antes de seguir |
|---|---|---|
| `notebooks/ingestao/02_silver.py` | `catalog`, `dataset=training` | `spark.table(f"{catalog}.silver.give_me_some_credit").count()` > 0. Se vier vazia com "[OK]" no log, apague o checkpoint em `/Volumes/{catalog}/silver/checkpoints/give_me_some_credit_training/` e rode de novo. |

---

## Fase 3 — Gold (star schema)

| Notebook | Parâmetros | Checar antes de seguir |
|---|---|---|
| `notebooks/ingestao/03_gold.py` | `catalog` | `gold.dim_customer` e `gold.fct_credit_profile` populadas (mesma contagem da Silver) |

---

## Fase 4 — EDA (diagnóstico de qualidade)

| Notebook | Parâmetros | O que olhar |
|---|---|---|
| `notebooks/ead/01_eda_bronze.py` | `catalog`, `dataset` | Completude, duplicidade por `customer_id`, cardinalidade |
| `notebooks/ead/02_eda_silver.py` | `catalog` | Distribuição do target, flags de qualidade (`is_monthly_income_null`, `has_delinquency_outlier`) |

```>>>EXECUTAR ALL```

Opcional — não bloqueia a Fase 5, mas ajuda a pegar problema de dado antes
de treinar modelo em cima dele.

---

## Fase 5 — Validação de hipóteses e perguntas

| Notebook | Parâmetros | Responde | Persiste em |
|---|---|---|---|
| `notebooks/ml/01_hipoteses_h1_h4.py` | `catalog` | H1, H2, H3, H4 (README) | `gold.gold_hypotheses_validation` |
| `notebooks/ml/02_vies_threshold_roi.py` | `catalog` | Q1, Q2, Q3, Q4 (README) | MLflow (`{catalog}.gold.credito_risco_xgb`, alias `champion`) |

Checagens específicas:
- ROC-AUC do XGBoost por volta de 0.85–0.87
- H3 (piso de aprovação) e Q2 (custo assimétrico) devem dar **thresholds
  diferentes** — é esperado, não é bug (ver `src/config/business_params.py`)
- Rodar `02_vies_threshold_roi.py` uma segunda vez deve carregar o modelo
  do Registry, não treinar de novo

---

## Em construção (roadmap, sem notebook ainda)

Já documentado como comentário em `resources/risco_credito_pipeline.yml`
— será adicionado aqui quando o notebook existir:

- **Inferência batch** (`notebooks/ml/03_inferencia_batch.py`) — aplica o
  modelo `@champion` sobre o dataset `scoring`
- **Monitoramento de drift de dados** (`notebooks/monitoracao/01_drift_dados.py`)
  — distribuição das features de entrada mudou vs. treino?
- **Monitoramento de concept drift** (`notebooks/monitoracao/02_concept_drift.py`)
  — a relação entre features e target mudou (performance real caindo)?
- **Retreino do modelo** — task agendada ou novo run de `01_hipoteses_h1_h4.py`
  / `02_vies_threshold_roi.py` com novo alias no Registry

---

## Execução automatizada (Job)

Todas as fases 1-5 (exceto o Reset, que é manual/destrutivo) já estão
encadeadas em `resources/risco_credito_pipeline.yml`:

```
databricks bundle deploy --target dev
databricks bundle run risco_credito_pipeline --target dev
```
