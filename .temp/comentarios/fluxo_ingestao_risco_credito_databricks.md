# Fase de ingestão de dados — Projeto `risco_credito_databricks`

> **Objetivo deste documento:** explicar, de forma didática, como os dados do projeto saem do arquivo CSV bruto e chegam a tabelas prontas para treinar e aplicar o modelo de risco de crédito.
> **Dataset:** *Give Me Some Credit* (Kaggle), cerca de 150 mil clientes, com a variável-alvo `SeriousDlqin2yrs` (o cliente teve atraso de 90+ dias em 2 anos?).
> **Plataforma:** Databricks + Unity Catalog, com **arquitetura medalhão** (Bronze → Silver → Gold).

---

## 1. A ideia em uma imagem

Pense na ingestão como uma **linha de refino**: o dado entra "sujo" e vai ficando mais confiável a cada camada.

```
  CSVs no Volume (raw_landing)
   ├── training/cs-training.csv   (tem o alvo)
   └── scoring/cs-test.csv        (sem o alvo)
             │
             ▼
  ┌────────────────────────────────────────────────────────────┐
  │ BRONZE   "guarde tudo como chegou"                         │
  │ Auto Loader (streaming incremental) + metadados            │
  │ credito_dev.bronze.give_me_some_credit_{dataset}_raw       │
  └────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌────────────────────────────────────────────────────────────┐
  │ SILVER   "limpe e padronize, sem jogar nada fora"          │
  │ Renomeia colunas, tipa, sinaliza problemas, MERGE          │
  │ credito_dev.silver.give_me_some_credit                     │
  │ credito_dev.silver.give_me_some_credit_scoring             │
  └────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌────────────────────────────────────────────────────────────┐
  │ GOLD     "entregue no vocabulário de negócio"              │
  │ Une treino + scoring, cria feature derivada, star schema   │
  │ credito_dev.gold.dim_customer                              │
  │ credito_dev.gold.fct_credit_profile                        │
  └────────────────────────────────────────────────────────────┘
             │
             ▼
  Modelagem, inferência em lote e monitoramento
```

---

## 2. Como o código está organizado

O projeto segue um padrão simples e muito usado no mercado:

| Onde | O que contém | Analogia |
|---|---|---|
| `src/` | **A lógica de verdade** (funções Python testáveis) | A cozinha: onde a comida é feita |
| `notebooks/` | **Só orquestração**: recebe parâmetros, importa e chama a função de `src/` | O garçom: só leva o pedido e entrega |

Na ingestão, temos três pares "notebook + módulo":

| Camada | Notebook (orquestra) | Módulo (lógica) | Função de entrada |
|---|---|---|---|
| Bronze | `notebooks/ingestao/01_bronze.py` | `src/ingestion/bronze.py` | `run_bronze_ingestion` |
| Silver | `notebooks/ingestao/02_silver.py` | `src/ingestion/silver.py` | `run_silver_ingestion` |
| Gold | `notebooks/ingestao/03_gold.ipynb` | `src/features/gold.py` | `run_gold_ingestion` |

**Por que separar assim?** Se a regra de limpeza precisar mudar, muda-se em **um único lugar** (`src/`). Os notebooks não reimplementam nada, e a análise exploratória (EDA) pode importar as mesmas funções.

### O parâmetro `catalog` (promoção entre ambientes)

Todos os notebooks recebem o **catálogo** por *widget* (`credito_dev`, `credito_hml`, `credito_prd`). Assim o mesmo código roda em desenvolvimento, homologação e produção **sem alterar uma linha**. O que muda é só o parâmetro.

Os notebooks de Bronze e Silver recebem também o widget `dataset`, com valores `training` ou `scoring`.

---

## 3. Camada Bronze — "guardar tudo como chegou"

**Arquivos:** `01_bronze.py` (notebook) e `bronze.py` (lógica).

### 3.1 De onde vêm os dados

Os CSVs ficam em um Volume do Unity Catalog, separados **por tipo de dataset**:

```
/Volumes/{catalog}/bronze/raw_landing/training/   → cs-training.csv (com alvo)
/Volumes/{catalog}/bronze/raw_landing/scoring/    → cs-test.csv     (sem alvo)
```

**Por que separar em duas pastas?** Os dois arquivos têm significados diferentes: o de *training* serve para treinar o modelo; o de *scoring* é o lote novo a ser pontuado. Em uma versão anterior, um glob genérico lia os dois juntos e **contaminou** a tabela Bronze. Hoje cada `dataset` gera **sua própria tabela Bronze**, e a função `run_bronze_ingestion` não tem valor padrão para `dataset`, justamente para obrigar quem executa a declarar qual está processando.

### 3.2 O que a Bronze faz, passo a passo

1. **Garante a infraestrutura** (`ensure_bronze_infra`): cria o schema `bronze` e os volumes `raw_landing`, `checkpoints` e `schemas`, se não existirem.
2. **Lê os CSVs com Auto Loader** (`cloudFiles`): é uma leitura em streaming *incremental*, ou seja, só processa arquivos novos, sem reler tudo a cada execução.
3. **Usa schema explícito** (`CREDIT_SCHEMA`): os tipos das 12 colunas são declarados no código. Isso evita o custo de inferir o schema e traz previsibilidade.
4. **Renomeia `_c0` para `customer_id`**: a primeira coluna do CSV é o índice, que vira o identificador do cliente.
5. **Adiciona colunas de linhagem** (rastreabilidade):
   - `_ingestion_timestamp`: quando o registro foi ingerido;
   - `_source_file`: de qual arquivo veio.
6. **Grava em Delta, modo `append`**, com `trigger(availableNow=True)`: processa tudo o que estiver disponível em um único lote e encerra. É barato e simples de agendar.
7. **Valida o resultado**: se a tabela estiver vazia após a carga, lança um `RuntimeError` com as causas mais comuns (nenhum CSV na pasta, ou checkpoint que já marcou os arquivos como processados).

**Tabela de saída:** `{catalog}.bronze.give_me_some_credit_{dataset}_raw`

> **Conceito-chave: checkpoint.** É a "memória" do streaming: um registro de até onde ele já leu. É ele que garante que um arquivo não seja processado duas vezes. Fica em `/Volumes/{catalog}/bronze/checkpoints/...`.

---

## 4. Camada Silver — "limpar e padronizar, sem jogar nada fora"

**Arquivos:** `02_silver.py` (notebook) e `silver.py` (lógica).

A `silver.py` é a **fonte única da verdade** para nomes de colunas e regras de limpeza. Tanto a ingestão quanto a EDA devem importar dela, nunca copiar as regras.

### 4.1 Entrada e saída

| `dataset` | Lê de | Grava em |
|---|---|---|
| `training` (padrão) | `bronze.give_me_some_credit_training_raw` | `silver.give_me_some_credit` |
| `scoring` | `bronze.give_me_some_credit_scoring_raw` | `silver.give_me_some_credit_scoring` |

### 4.2 Renomeação de colunas (`COLUMN_MAPPING`)

Os nomes originais do Kaggle viram nomes descritivos em *snake_case*:

| Bronze (Kaggle) | Silver |
|---|---|
| `SeriousDlqin2yrs` | `target_default_2yrs` |
| `RevolvingUtilizationOfUnsecuredLines` | `revolving_utilization_unsecured` |
| `age` | `age` |
| `NumberOfTime30-59DaysPastDueNotWorse` | `num_times_30_59_days_late` |
| `DebtRatio` | `debt_ratio` |
| `MonthlyIncome` | `monthly_income` |
| `NumberOfOpenCreditLinesAndLoans` | `num_open_credit_lines_and_loans` |
| `NumberOfTimes90DaysLate` | `num_times_90_days_late` |
| `NumberRealEstateLoansOrLines` | `num_real_estate_loans_or_lines` |
| `NumberOfTime60-89DaysPastDueNotWorse` | `num_times_60_89_days_late` |
| `NumberOfDependents` | `num_dependents` |

### 4.3 Regras de qualidade: a "quarentena lógica"

A filosofia é **não descartar linhas silenciosamente**. Em vez de dar `.filter()` e perder registros, a Silver **marca** o problema ou **anula** o valor inválido. Assim o registro continua disponível para auditoria e a contagem de linhas Bronze → Silver não é distorcida.

| Regra | O que acontece |
|---|---|
| `customer_id` nulo | A linha é removida (sem chave não há como identificar o cliente) |
| `age` fora de **18–120** | O valor vira `null` (a linha permanece) |
| `monthly_income` nulo | Cria a flag `is_monthly_income_null = 1` |
| `num_dependents` nulo | Vira `0` e é convertido para inteiro |
| Atraso (30-59, 60-89 ou 90+) com valor **≥ 96** | Cria a flag `has_delinquency_outlier = 1` (96 e 98 são códigos de erro conhecidos do dataset) |
| Toda linha | Recebe `_silver_processed_at` (timestamp de processamento) |

### 4.3.1 Como a gravação funciona (MERGE idempotente)

A Silver usa `foreachBatch`, que trata cada micro-lote com a função `upsert_to_silver`:

1. **Deduplica** o lote por `customer_id`, mantendo o registro com `_ingestion_timestamp` mais recente.
2. Se a tabela Silver **ainda não existe**, cria por `overwrite`.
3. Se já existe, faz **MERGE** por `customer_id`: atualiza quem já existe (`whenMatchedUpdateAll`) e insere quem é novo (`whenNotMatchedInsertAll`).

> **Idempotente** significa que rodar duas vezes com os mesmos dados dá o mesmo resultado, sem duplicar linhas.

### 4.4 Proteções contra falha

- O **volume de checkpoint é criado antes** do `writeStream` (isso corrigiu o erro `UC_VOLUME_NOT_FOUND`).
- **Validação pós-carga:** se a tabela Silver não existir ao final, lança `RuntimeError`. Isso cobre o cenário do *retry sem dado novo*, em que o checkpoint já consumiu toda a Bronze, **zero micro-lotes rodam** e o streaming termina "com sucesso" sem criar nada. Sem essa checagem, o erro só apareceria depois, como `TABLE_OR_VIEW_NOT_FOUND` no notebook seguinte.

---

## 5. Camada Gold — "entregar no vocabulário de negócio"

**Arquivos:** `03_gold.ipynb` (notebook) e `gold.py` (lógica).

A Gold é construída em **batch** (não streaming). A Silver já cuida da incrementalidade, e a Gold é só uma projeção derivada dela. Por isso, a cada execução ela é **recalculada por inteiro** com `overwrite`.

### 5.1 Treino e scoring na mesma Gold

A função `_read_silver_combined` **une** a Silver de treino com a de scoring (`unionByName`):

- As linhas de **scoring** ficam com `target_dlq_2yrs` **nulo por construção** (é o lote sem resposta conhecida).
- É esse nulo que os notebooks de **inferência em lote** e de **monitoramento** usam para identificar "o que é lote novo a pontuar".
- Se a Silver de scoring **ainda não existir**, a Gold é construída só com o treino e emite um `[AVISO]`. Essa verificação usa `spark.catalog.tableExists`.
- Se os schemas das duas Silver divergirem **além do alvo**, a Gold falha alto com `RuntimeError`, pois é um bug de upstream e a Gold não deve tentar adivinhar como conciliar.

### 5.2 As duas tabelas de saída (star schema)

**`gold.dim_customer`** — dimensão de cliente (atributos demográficos):

| Coluna | Descrição |
|---|---|
| `customer_id` | Chave do cliente (deduplicada) |
| `age` | Idade |
| `num_dependents` | Número de dependentes |
| `_gold_processed_at` | Timestamp de processamento |

**`gold.fct_credit_profile`** — fato de perfil de crédito:

| Coluna | Origem |
|---|---|
| `customer_id` | Chave |
| `monthly_income` | Silver |
| `debt_ratio` | Silver |
| `revolving_utilization` | Silver (`revolving_utilization_unsecured`) |
| `num_open_credit_lines` | Silver (`num_open_credit_lines_and_loans`) |
| `num_real_estate_loans` | Silver (`num_real_estate_loans_or_lines`) |
| `num_times_30_59_days_late` | Silver |
| `num_times_60_89_days_late` | Silver |
| `num_times_90_days_late` | Silver |
| `total_delinquency_events` | **Derivada:** soma das três colunas de atraso |
| `target_dlq_2yrs` | Silver (`target_default_2yrs`). **Nulo nas linhas de scoring** |
| `_gold_processed_at` | Timestamp de processamento |

### 5.3 Por que a Gold renomeia colunas de novo?

Os notebooks de análise e modelagem já existentes usam nomes **curtos** (`target_dlq_2yrs`, `revolving_utilization`...). Em vez de reescrever esses notebooks ou renomear a Silver, a Gold faz o **de-para**. Esse é o papel dela na arquitetura medalhão: entregar o dado no vocabulário que quem consome já espera.

| Silver | Gold |
|---|---|
| `target_default_2yrs` | `target_dlq_2yrs` |
| `revolving_utilization_unsecured` | `revolving_utilization` |
| `num_open_credit_lines_and_loans` | `num_open_credit_lines` |
| `num_real_estate_loans_or_lines` | `num_real_estate_loans` |

### 5.4 Decisões de design importantes

- **Feature engineering na Gold, não na Silver.** A Silver só limpa e padroniza; `total_delinquency_events` é uma feature e por isso nasce na Gold.
- **As flags de qualidade não vão para a Gold** (`is_monthly_income_null`, `has_delinquency_outlier`). Os notebooks de modelagem tratam como *feature* qualquer coluna que não esteja na lista de ignoradas. Se as flags fossem para a Gold, virariam feature **sem ninguém ter decidido isso conscientemente**.
- **Validação final:** se `dim_customer` ou `fct_credit_profile` ficarem vazias, a execução falha. Ao final, imprime a contagem de linhas e quantas têm alvo nulo.

---

## 6. Como executar

### 6.1 Ordem obrigatória

```
1) 01_bronze  → dataset = training
2) 01_bronze  → dataset = scoring        (opcional, se houver lote a pontuar)
3) 02_silver  → dataset = training
4) 02_silver  → dataset = scoring        (opcional)
5) 03_gold                               (sempre por último)
```

Cada `dataset` exige uma execução própria dos notebooks de Bronze e Silver. Em produção, isso pode ser feito com **duas tasks no mesmo Job**, cada uma com um valor de `dataset`.

### 6.2 Desenvolvimento x produção

- **Desenvolvimento:** roda-se os notebooks manualmente. O `uv sync` (instalação de dependências) aparece nos notebooks **apenas para uso em desenvolvimento**.
- **Produção:** a execução é via **Databricks Job** (`job_ingestao`, com as tasks `01_bronze`, `02_silver` e `03_gold`), com a configuração em `resources/job.yml`. Os notebooks ficam limpos de qualquer código de gerenciamento de ambiente. Há inclusive um script de CI (`scripts/check_notebook_hygiene.py`) que bloqueia commits com `%uv sync` ou `%pip install` fora de comentários.

### 6.3 Checagem rápida

Cada notebook termina exibindo as 10 primeiras linhas da tabela gerada (`display(spark.table(...).limit(10))`), para confirmação visual.

---

## 7. Problemas já encontrados e como foram resolvidos

Esta seção é uma "memória de projeto": conhecer estes casos evita repetir os mesmos erros.

| Sintoma | Causa | Solução |
|---|---|---|
| Tabela Bronze com dados misturados | Um glob lia `cs-training.csv` e `cs-test.csv` juntos | Subpastas `training/` e `scoring/`, com uma tabela Bronze por dataset |
| `UC_VOLUME_NOT_FOUND` na Silver | O checkpoint era referenciado antes de o volume existir | Criar o volume antes do `writeStream` |
| Silver "OK" mas com zero linhas / `TABLE_OR_VIEW_NOT_FOUND` depois | Checkpoint antigo já havia consumido a Bronze; `availableNow` não roda nenhum micro-lote no retry | Validação pós-carga com `RuntimeError`. Para corrigir: **apagar o checkpoint** (`/Volumes/credito_dev/silver/checkpoints/give_me_some_credit_training/`) e rodar de novo |
| Gold falhando por `silver.give_me_some_credit_scoring` inexistente | `try/except` em volta de `spark.table()` não captura o erro em Serverless (Spark Connect), pois o schema é resolvido de forma *lazy* | Trocar por `spark.catalog.tableExists(...)`, que é uma chamada de catálogo confiável |
| Erro de shell no notebook Silver | `%sh` usado como magic de linha em vez de `%%sh` (magic de célula, que deve ficar na 1ª linha) | Usar `%%sh` na primeira linha da célula |
| Correção do `gold.py` "não pegava" no Databricks | Painel de *Changes* do Git folder com estado desatualizado ("no outstanding changes") | Conferir com `git status` e fazer o commit pelo Web Terminal |
| Constantes financeiras divergentes entre módulos | Valores diferentes usados em análises distintas | Centralizadas em `src/config/business_params.py` |

---

## 8. Pontos de atenção (para revisar)

1. **Bug latente na mensagem de erro da Gold.** Em `run_gold_ingestion`, a mensagem do `RuntimeError` para Gold vazia referencia `silver_table`, variável que **não existe** nessa função (os parâmetros são `silver_table_name` e `scoring_table_name`). Se a Gold realmente ficar vazia, o usuário verá um `NameError` em vez da mensagem pretendida. Sugestão: trocar por `{catalog}.{silver_schema}.{silver_table_name}`.
2. **Docstring desatualizada.** A Gold cita "as 10 variáveis explicativas", mas `fct_credit_profile` traz 8 colunas de origem, mais a derivada `total_delinquency_events`. As outras duas variáveis (`age` e `num_dependents`) estão em `dim_customer`.
3. **Faixa de idade.** O código atual usa **18–120**. Versões anteriores do `silver.py` (e o dicionário de dados de 09/09) citavam 18–115. Vale confirmar que o dicionário de dados reflete a regra vigente.
4. **Padrão de `uv sync` inconsistente entre notebooks.** No `02_silver.py` ele está como célula `%%sh` ativa; no `01_bronze.py` e no `03_gold.ipynb` está comentado. Convém padronizar, conforme a decisão de manter os notebooks limpos.

---

## 9. Glossário rápido

| Termo | Significado simples |
|---|---|
| **Arquitetura medalhão** | Organização em camadas (Bronze, Silver, Gold), cada uma mais refinada que a anterior |
| **Unity Catalog** | Governança do Databricks: catálogo → schema → tabela/volume |
| **Volume** | Pasta gerenciada dentro do Unity Catalog, usada para arquivos (CSVs, checkpoints) |
| **Auto Loader (`cloudFiles`)** | Leitor incremental de arquivos: só processa o que é novo |
| **Checkpoint** | Registro do progresso do streaming, evita reprocessar |
| **`trigger(availableNow=True)`** | Processa tudo o que está pendente e encerra |
| **MERGE / upsert** | Atualiza registros existentes e insere os novos, em uma só operação |
| **Idempotência** | Rodar de novo produz o mesmo resultado, sem duplicar |
| **Quarentena lógica** | Marcar/anular o dado inválido em vez de apagar a linha |
| **Star schema** | Modelagem com uma tabela fato (métricas) ligada a dimensões (atributos) |
| **Widget** | Parâmetro de entrada de um notebook Databricks |
