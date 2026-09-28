# Como criar o job de scoring (teste) — Risco de Crédito

Passo a passo para criar, na UI do Databricks, o job que roda o pipeline
com o lote de **scoring** (`cs-test.csv`, holdout sem rótulo), separado
do job que já existe para o lote de **treino**.

> **Pré-requisito.** Esta pipeline depende do job de treino já ter
> rodado antes — é ele quem registra o `@champion` no Model Registry.
> Sem isso, a task de inferência falha ao carregar o modelo. Rode
> sempre o job de treino primeiro, e só depois este de scoring.

---

## 1. Criar o job e o parâmetro `catalog`

Em **Jobs & Pipelines > Create Job**, dê um nome como
`risco-credito - pipeline scoring`. Em **Job parameters**, adicione
`catalog = credito_dev` — assim toda task referencia
`{{job.parameters.catalog}}` em vez de repetir o valor em cada uma.

---

## 2. Task 1 — Bronze do lote scoring

| Campo | Valor |
|---|---|
| Notebook | `notebooks/ingestao/01_bronze.py` |
| Parâmetros | `catalog = {{job.parameters.catalog}}`, `dataset = scoring` |
| Compute | Serverless |
| Depends on | — (primeira task) |

---

## 3. Task 2 — Silver do lote scoring

| Campo | Valor |
|---|---|
| Notebook | `notebooks/ingestao/02_silver.py` |
| Parâmetros | `catalog`, `dataset = scoring` |
| Depends on | Task 1 |

**Atenção:** no roteiro do projeto essa task (`silver_scoring`) ainda
está marcada como "proposta, não confirmada" no YAML do bundle. Sem
ela, o Bronze de scoring fica órfão e a inferência não encontra dado
novo para pontuar.

---

## 4. Task 3 — Gold

| Campo | Valor |
|---|---|
| Notebook | `notebooks/ingestao/03_gold.ipynb` (é um `.ipynb`, não `.py`) |
| Parâmetros | `catalog` |
| Depends on | Task 2 |

O Gold é o mesmo notebook do treino — ele une treino e scoring na
mesma tabela, então essa execução atualiza a Gold já existente com os
novos registros de scoring.

---

## 5. Task 4 — Inferência batch

| Campo | Valor |
|---|---|
| Notebook | `notebooks/ml/03_inferencia_batch.py` |
| Parâmetros | `catalog`, `threshold_aprovacao = 0,45` |
| Depends on | Task 3 |

O `threshold_aprovacao = 0,45` é o corte operacional definido em H3 —
o default do widget no notebook é `0,56`, então vale sobrescrever
explicitamente.

**Pré-requisito fora deste job:** o job de treino precisa ter rodado
ao menos uma vez antes, registrando o `@champion` no Model Registry —
sem isso, esta task falha ao tentar carregar o modelo.

---

## 6. Task 5 — Data drift

| Campo | Valor |
|---|---|
| Notebook | `notebooks/monitoracao/01_drift_dados.py` |
| Parâmetros | `catalog` |
| Depends on | Task 4 |

Usa o lote recém-pontuado pela inferência como "lote atual" para
comparar com o treino.

---

## 7. Task 6 — Concept drift

| Campo | Valor |
|---|---|
| Notebook | `notebooks/monitoracao/02_concept_drift.py` |
| Parâmetros | `catalog`, `simulation_mode = true` |
| Depends on | Task 3 (Gold) |

Independente da inferência — não há rótulo real ainda, então roda em
modo simulado. Pode ficar em paralelo à Task 5 se preferir.

---

## 8. Rodar e, se fizer sentido, agendar

Clique em **Run now** e acompanhe as 6 tasks no grafo, do mesmo jeito
que no job de treino. Se quiser recorrência, configure o **Schedule**
só depois de validar uma execução manual completa — e programe o
horário sempre depois do job de treino, nunca antes, já que esta
pipeline depende do `@champion` mais recente.

---

## Resumo do encadeamento

```
Bronze (scoring)
     │
Silver (scoring)
     │
   Gold
   ├────────────────┐
Inferência batch   Concept drift (simulado)
     │
Data drift
```
