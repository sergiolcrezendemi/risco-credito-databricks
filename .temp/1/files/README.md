# Risco de Crédito — Score de Probabilidade de Inadimplência

> Pipeline ponta a ponta no **Databricks**: ingestão em arquitetura medalhão (Bronze → Silver → Gold), modelo **XGBoost** registrado no Unity Catalog, explicabilidade com **SHAP**, decisão de aprovação por **custo assimétrico**, análise de viés e **monitoramento de drift**.

![CI](https://github.com/SEU-USUARIO/risco-credito-databricks/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Databricks](https://img.shields.io/badge/platform-Databricks-FF3621.svg)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)

*Parte da série **Databricks de Ponta a Ponta**.*

---

## Sumário

- [Resultados em destaque](#resultados-em-destaque)
- [Problema de negócio](#problema-de-negócio)
- [Base de dados](#base-de-dados)
- [Arquitetura](#arquitetura)
- [Stack](#stack)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Modelagem](#modelagem)
- [Hipóteses e perguntas de investigação](#hipóteses-e-perguntas-de-investigação)
- [Resultados detalhados](#resultados-detalhados)
- [Monitoramento](#monitoramento)
- [Boas práticas de engenharia](#boas-práticas-de-engenharia)
- [Como executar](#como-executar)
- [Limitações assumidas](#limitações-assumidas)
- [Status e roadmap](#status-e-roadmap)
- [Documentação adicional](#documentação-adicional)
- [Autor](#autor)

---

## Resultados em destaque

| Indicador | Resultado |
|---|---|
| **AUC-ROC** (conjunto de teste) | **0,8641** (meta do projeto: ≥ 0,85) |
| **Threshold ótimo** (custo assimétrico) | **0,56**, contra 0,50 do corte ingênuo |
| **Inadimplência da carteira aprovada** | 6,69% sem modelo → **2,35%** com modelo |
| **Impacto financeiro líquido estimado** | **R$ 13,45 milhões** em 37.500 operações de teste |
| **Viés identificado** | Jovens (18–30) com mais bons pagadores negados; 60+ e renda alta com mais maus pagadores aprovados |

> **Atenção:** os valores em R$ usam parâmetros **ilustrativos** (R$ 15.000 por mau pagador aprovado e R$ 1.500 por bom pagador negado). Servem para demonstrar a metodologia, não como projeção de negócio. Detalhes em [Resultados detalhados](#resultados-detalhados).

---

## Problema de negócio

Uma instituição financeira precisa decidir, para cada solicitação de crédito, se aprova ou não a operação. A decisão precisa ser **rápida**, **consistente entre analistas** e **defensável em auditoria**.

O objetivo deste projeto é construir um score de probabilidade de inadimplência que sustente essa decisão de ponta a ponta: **da ingestão do dado até uma decisão de aprovação rastreável**, com monitoramento após a entrada em uso.

### Métricas

| Tipo | Métrica |
|---|---|
| **Primária** | AUC-ROC ≥ 0,85 no conjunto de teste (soluções de topo da competição original do Kaggle atingem cerca de 0,86–0,87) |
| **Complementares** | KS-statistic (separação entre bons e maus pagadores); recall na classe inadimplente a um nível de precisão definido pelo negócio |
| **De negócio** | Redução estimada de inadimplência em R$, comparando "com modelo" e a política atual (aprovação sem score) |

---

## Base de dados

| Fonte | Uso |
|---|---|
| [**Give Me Some Credit**](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle) | Base principal: cerca de 150.000 clientes pessoa física, 10 variáveis explicativas e o alvo `SeriousDlqin2yrs` (1 = atraso de 90+ dias nos próximos 2 anos) |
| [**SCR.data**](https://dadosabertos.bcb.gov.br/dataset/scr_data) (BACEN, dados agregados) | Referência externa para contextualizar a taxa de inadimplência por segmento. **Não** é usada para treino |

### Variáveis (nomes na Gold)

| Variável | Descrição |
|---|---|
| `revolving_utilization` | Saldo em cartões e linhas pessoais dividido pelos limites |
| `age` | Idade do cliente |
| `num_dependents` | Número de dependentes |
| `debt_ratio` | Pagamentos mensais de dívidas divididos pela renda bruta mensal |
| `monthly_income` | Renda mensal |
| `num_open_credit_lines` | Empréstimos e linhas de crédito abertos |
| `num_real_estate_loans` | Financiamentos e linhas imobiliárias |
| `num_times_30_59_days_late`, `num_times_60_89_days_late`, `num_times_90_days_late` | Vezes com atraso em cada faixa |
| `total_delinquency_events` | Feature derivada: soma dos três atrasos |
| `target_dlq_2yrs` | Alvo (nulo nas linhas de scoring, por construção) |

---

## Arquitetura

```
 CSVs no Volume (raw_landing/training e raw_landing/scoring)
                     │
                     ▼
   BRONZE   Auto Loader (streaming incremental), schema explícito, linhagem
                     │
                     ▼
   SILVER   Padronização de schema, quarentena lógica, MERGE idempotente
                     │
                     ▼
   GOLD     Star schema: dim_customer + fct_credit_profile (treino + scoring)
                     │
      ┌──────────────┼───────────────────────────┐
      ▼              ▼                           ▼
   Treino e      Inferência em lote        Monitoramento
   validação     (@champion)               (data drift e concept drift)
   MLflow + SHAP + Unity Catalog Model Registry
```

| Camada | Responsabilidade | Tabela de saída |
|---|---|---|
| **Bronze** | Ingestão incremental com Auto Loader; uma tabela por dataset | `bronze.give_me_some_credit_{training\|scoring}_raw` |
| **Silver** | Renomeação, tipagem e regras de qualidade sem descartar linhas (idade fora de 18–120 vira nulo; flags para renda nula e códigos de erro ≥ 96); MERGE por `customer_id` | `silver.give_me_some_credit` e `silver.give_me_some_credit_scoring` |
| **Gold** | Feature derivada, renomeação para o vocabulário de negócio e união treino + scoring | `gold.dim_customer` e `gold.fct_credit_profile` |
| **Modelo** | XGBoost registrado com o alias `@champion` | `gold.credito_risco_xgb` |
| **Inferência** | Pontua o lote sem alvo e grava probabilidade e decisão | `gold.credito_score_predictions` |
| **Monitoramento** | PSI/KS por variável e queda de AUC | `ml.monitoramento_drift_dados` e `ml.monitoramento_concept_drift` |

> O catálogo (`credito_dev`, `credito_hml`, `credito_prd`) é um **parâmetro** dos notebooks. A promoção entre ambientes não exige mudança de código.

---

## Stack

| Área | Tecnologias |
|---|---|
| Plataforma | Databricks (Serverless), Unity Catalog, Delta Lake, Databricks Jobs |
| Ingestão | PySpark, Auto Loader (`cloudFiles`), Structured Streaming |
| Modelagem | XGBoost, Regressão Logística (scikit-learn), SHAP |
| MLOps | MLflow Tracking e Model Registry (aliases no Unity Catalog) |
| Qualidade | pytest, Ruff, pre-commit, GitHub Actions |
| Dependências | `pyproject.toml` + `uv` |

---

## Estrutura do repositório

```
risco-credito-databricks/
├── notebooks/
│   ├── ingestao/       # 01_bronze, 02_silver, 03_gold  (só orquestração)
│   ├── ead/            # 01_eda_bronze, 02_eda_silver
│   ├── ml/             # 01_hipoteses_h1_h4, 02_vies_threshold_roi, 03_inferencia_batch
│   └── monitoracao/    # 01_drift_dados, 02_concept_drift
├── src/                # lógica de negócio (funções testáveis)
│   ├── config/         # business_params.py (parâmetros financeiros)
│   ├── ingestion/      # bronze.py, silver.py, maintenance.py
│   ├── features/       # gold.py
│   ├── quality/        # profiling.py
│   ├── models/         # hypothesis_validation.py, bias_roi_validation.py
│   ├── ml/             # decisao.py
│   └── monitoring/     # data_drift.py, concept_drift.py
├── tests/              # testes unitários (pytest), sem necessidade de cluster
├── resources/          # definição dos Jobs (job.yml)
├── scripts/            # check_notebook_hygiene.py
├── docs/               # documentação didática de cada fase
├── pyproject.toml
└── README.md
```

**Princípio central:** a lógica vive em `src/` (funções puras e testáveis); os notebooks apenas recebem parâmetros, chamam essas funções e exibem os resultados.

---

## Modelagem

| Aspecto | Decisão |
|---|---|
| **Modelo principal** | **XGBoost**: dado tabular, alvo binário desbalanceado (~6,7% de positivos) e interações não lineares entre variáveis financeiras |
| **Modelo de referência** | **Regressão Logística**: piso de desempenho e leitura direta por coeficientes |
| **Explicabilidade** | **SHAP** para o modelo principal: cada decisão pode ser rastreada até as variáveis que mais pesaram |
| **Desbalanceamento** | Avaliação por AUC-ROC e KS, e threshold definido por **custo assimétrico** (aprovar um mau pagador custa mais que negar um bom) |
| **Decisão** | Probabilidade ≥ threshold → `negado`; caso contrário → `aprovado` |
| **Versionamento** | Modelo `credito_risco_xgb` no Unity Catalog com o alias `@champion`; cada previsão grava versão e threshold usados |

---

## Hipóteses e perguntas de investigação

O projeto separa **hipóteses** (teses escritas antes do teste, com critério de confirmação) de **perguntas** (dúvidas respondidas com dados). Cada uma sustenta uma decisão.

### Hipóteses

| # | Hipótese | Camada | Como é testada |
|---|---|---|---|
| **H1** | Clientes com maior utilização de crédito rotativo (`revolving_utilization`) têm probabilidade significativamente maior de inadimplência, pois uso próximo do limite indica menor folga financeira | Confiança no modelo | Correlação entre o valor da variável e o seu SHAP value |
| **H2** | O `debt_ratio` tem mais poder preditivo que a `age` isoladamente: a capacidade de pagamento pesa mais que o perfil demográfico | Confiança no modelo | Comparação do \|SHAP\| médio, com margem relativa mínima de 10% |
| **H3** | Existe um threshold que reduz a inadimplência da carteira aprovada sem derrubar a aprovação abaixo de um piso viável | Decisão operacional | Varredura de thresholds com piso de aprovação de 70% (ilustrativo) |
| **H4** | O modelo teria evitado mais perdas em R$ do que custaria rejeitar bons pagadores | Decisão operacional | Perda evitada − custo de oportunidade, no threshold de H3 |

### Perguntas

| # | Pergunta | Decisão que sustenta |
|---|---|---|
| **Q1** | Quais variáveis têm maior poder preditivo (SHAP)? | Confiar ou não no modelo |
| **Q2** | Qual o threshold ótimo sob custo assimétrico? | Regra operacional de aprovação |
| **Q3** | O desempenho é estável por faixa de idade e renda? | Ir ou não para produção (risco regulatório) |
| **Q4** | Qual o impacto financeiro líquido vs. política atual? | Aprovar o investimento |

> A justificativa completa de cada hipótese está em [`docs/justificativa_hipoteses.md`](docs/justificativa_hipoteses.md).

---

## Resultados detalhados

Os números abaixo vêm de uma execução sobre a base de teste. Os valores em R$ são **ilustrativos**.

### Q1 — Importância das variáveis (SHAP)

| Posição | Variável |
|---|---|
| 1º | `total_delinquency_events` |
| 2º | `revolving_utilization` |
| 3º | `age` |
| 4º | `num_open_credit_lines` |
| 5º | `monthly_income` |

- O modelo prefere a variável **agregada** de atrasos às versões separadas (30–59, 60–89, 90+): a soma concentra o mesmo sinal.
- O uso do limite rotativo (2º) pesa bem mais que a renda (5º).
- **Ponto de atenção para H2:** `age` aparece em 3º, enquanto `debt_ratio` não aparece no top-5.

### Q2 — Threshold ótimo

| Threshold | Custo total esperado |
|---|---|
| 0,50 (padrão) | R$ 18,98 milhões |
| **0,56 (ótimo)** | **R$ 18,77 milhões** |

**Economia de cerca de R$ 204 mil** só pelo ajuste do corte. Como aprovar um mau pagador custa 10× mais que negar um bom, o threshold ótimo sobe um pouco: é um ajuste fino.

### Q3 — Viés por idade e renda

| Grupo | Alerta | Taxa |
|---|---|---|
| 18–30 anos | Bons pagadores negados indevidamente (FPR) | **29,7%** |
| 60+ anos | Maus pagadores aprovados por engano (FNR) | **48,3%** |
| Renda acima de R$ 8.250 | Maus pagadores aprovados por engano (FNR) | **42,5%** |

O modelo **não é neutro entre grupos**: é mais restritivo com jovens e mais permissivo com idosos e rendas altas. Uma hipótese para o caso dos jovens é o histórico de crédito mais curto, mas isso não foi testado. Há impacto tanto na carteira (risco não detectado concentrado em 60+ e renda alta) quanto regulatório e reputacional.

### Q4 — Impacto financeiro

| Cenário | Inadimplência | Perda estimada |
|---|---|---|
| Sem modelo (aprova todos) | 6,69% | R$ 30,08 milhões |
| Com modelo (threshold ótimo) | 2,35% | R$ 8,57 milhões |

- Redução bruta de perdas: **R$ 21,52 milhões**
- Custo de oportunidade (5.375 bons pagadores negados): **R$ 8,06 milhões**
- **Impacto líquido estimado: R$ 13,45 milhões** em 37.500 operações

**Conclusão:** o modelo é tecnicamente sólido e financeiramente favorável nas premissas adotadas. Antes de ir para produção, o viés contra jovens e a fuga de maus pagadores em renda alta devem ser tratados, possivelmente com calibração de threshold por segmento.

> Análise completa em [`docs/resultados_detalhados.md`](docs/resultados_detalhados.md).

---

## Monitoramento

| Notebook | O que mede | Métrica e limites |
|---|---|---|
| `01_drift_dados` | **Data drift**: os dados novos parecem com os de treino? | **PSI** por variável (`< 0,10` estável; `0,10–0,25` moderado; `≥ 0,25` severo) e **KS** |
| `02_concept_drift` | **Concept drift**: o modelo continua acertando? | Queda de **AUC-ROC** vs. referência (alerta a partir de 0,03) e KS |

- O alvo tem horizonte de **24 meses**, então o resultado real demora a chegar. O concept drift roda em **modo simulação** (lote sintético com relação variável-alvo enfraquecida) para validar a lógica de alerta, e alterna para dados reais quando a tabela de resultados realizados existir.
- Os resultados são gravados em tabelas de histórico e no MLflow (`/Shared/credito_risco_monitoramento`).

---

## Boas práticas de engenharia

- **Separação de responsabilidades:** lógica em `src/`, notebooks só orquestram.
- **Parametrização por ambiente:** catálogo por *widget*, sem alteração de código entre dev, hml e prd.
- **Ingestão idempotente:** MERGE por chave de negócio e checkpoint de streaming.
- **Falha alta e clara:** validações pós-carga com `RuntimeError` descritivo (tabela vazia, checkpoint desatualizado, schema divergente).
- **Quarentena lógica:** o dado inválido é marcado ou anulado, sem descartar linhas silenciosamente.
- **Fonte única de verdade:** mapeamento de colunas em `silver.py` e parâmetros financeiros em `business_params.py`.
- **Lógica testável sem cluster:** funções puras cobertas por `pytest`.
- **CI:** Ruff, pre-commit e testes no GitHub Actions; `check_notebook_hygiene.py` bloqueia comandos de ambiente (`%pip`, `%uv sync`) em notebooks versionados.
- **Rastreabilidade:** cada previsão grava versão do modelo, threshold e data/hora de execução.

---

## Como executar

### Pré-requisitos
- Workspace Databricks com Unity Catalog e um catálogo (por padrão, `credito_dev`)
- CSVs do Kaggle em `/Volumes/<catalogo>/bronze/raw_landing/training/` (`cs-training.csv`) e `.../scoring/` (`cs-test.csv`)

### Ordem de execução

| # | Notebook | Parâmetros |
|---|---|---|
| 1 | `ingestao/01_bronze` | `dataset = training` e depois `scoring` |
| 2 | `ingestao/02_silver` | `dataset = training` e depois `scoring` |
| 3 | `ingestao/03_gold` | `catalog` |
| 4 | `ead/01_eda_bronze` e `ead/02_eda_silver` | Diagnóstico (opcional) |
| 5 | `ml/01_hipoteses_h1_h4` | Valida H1–H4 |
| 6 | `ml/02_vies_threshold_roi` | Responde Q1–Q4 e registra o `@champion` |
| 7 | `ml/03_inferencia_batch` | `threshold_aprovacao` (padrão 0,56) |
| 8 | `monitoracao/01_drift_dados` e `02_concept_drift` | Monitoramento |

Em produção, a ingestão roda como **Databricks Job** (`resources/job.yml`), com as tarefas Bronze, Silver e Gold encadeadas.

### Desenvolvimento local

```bash
uv sync                      # instala dependências do pyproject.toml
pre-commit install           # ativa os hooks de qualidade
pytest                       # testes unitários
```

---

## Limitações assumidas

- **Dado antigo e de outro mercado:** a base é de 2011 e de clientes americanos. É usada como base **metodológica**, não como fonte de verdade sobre o mercado brasileiro. O cruzamento com o SCR.data contextualiza essa diferença, mas não a corrige.
- **Horizonte fixo:** o alvo tem 24 meses. Decisões com outro prazo exigem reavaliação do problema.
- **Valores financeiros ilustrativos:** todo resultado em R$ depende de parâmetros que ainda não vieram de um negócio real.
- **Concept drift simulado:** sem resultados reais atrasados, o alerta é validado com dados sintéticos.
- **Viés identificado, ainda não mitigado:** os desvios por idade e renda (Q3) foram medidos, mas não corrigidos.
- **Threshold definido na base de teste:** o corte é escolhido e avaliado no mesmo conjunto, o que tende a deixar o ganho ligeiramente otimista.

---

## Status e roadmap

| Item | Situação |
|---|---|
| Ingestão Bronze, Silver e Gold | ✅ Concluído |
| Diagnóstico (EAD) | ✅ Concluído |
| Hipóteses H1–H4 e perguntas Q1–Q4 | ✅ Concluído |
| Modelo registrado no Unity Catalog (`@champion`) | ✅ Concluído |
| Inferência em lote | ✅ Concluído |
| Monitoramento de data drift e concept drift | ✅ Implementado (concept drift em simulação) |
| CI (lint e testes) | ✅ Concluído |
| Feature Store versionada | ⏳ Planejado |
| Model Serving (endpoint em tempo real) | ⏳ Planejado |
| Lakehouse Monitoring com alertas automáticos | ⏳ Planejado |
| Tabela de resultados realizados (concept drift real) | ⏳ Planejado |
| Mitigação do viés por segmento | ⏳ Planejado |
| RBAC no Unity Catalog | ⏳ Planejado |

---

## Documentação adicional

Guias didáticos de cada fase, em `docs/`:

| Documento | Conteúdo |
|---|---|
| [`fluxo_ingestao_risco_credito_databricks.md`](docs/fluxo_ingestao_risco_credito_databricks.md) | Bronze, Silver e Gold |
| [`fase_ead_risco_credito_databricks.md`](docs/fase_ead_risco_credito_databricks.md) | Análise exploratória |
| [`fase_hipoteses_perguntas_risco_credito_databricks.md`](docs/fase_hipoteses_perguntas_risco_credito_databricks.md) | Hipóteses H1–H4 e perguntas Q1–Q4 |
| [`fase_monitoracao_risco_credito_databricks.md`](docs/fase_monitoracao_risco_credito_databricks.md) | Data drift e concept drift |
| [`justificativa_hipoteses.md`](docs/justificativa_hipoteses.md) | Por que essas quatro hipóteses |
| [`resultados_detalhados.md`](docs/resultados_detalhados.md) | Interpretação completa de Q1–Q4 |

---

## Autor

**Seu Nome** — Cientista de Dados
[LinkedIn](https://www.linkedin.com/in/SEU-PERFIL) · [GitHub](https://github.com/SEU-USUARIO)

## Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE`.
