# Fase de EAD (Análise Exploratória de Dados) — Projeto `risco_credito_databricks`

> **Objetivo deste documento:** explicar, de forma didática, como funciona a análise exploratória do projeto: o que os notebooks fazem, o que cada relatório mostra, como interpretar a saída e como a EAD se conecta às camadas Bronze e Silver.
> **Dataset:** *Give Me Some Credit* (Kaggle). Alvo: `target_default_2yrs` na Silver (`SeriousDlqin2yrs` na Bronze).
> **Nota sobre a sigla:** o projeto usa **EAD** (Exploratória/Análise de Dados) na pasta `notebooks/ead/`, e **EDA** (o termo em inglês) no nome dos arquivos. É a mesma coisa.

---

## 1. A ideia em uma imagem

A EAD é o **exame de saúde dos dados**. Ela **não altera nada**: só olha, mede e reporta.

```
   Bronze (dado como chegou)              Silver (dado tratado)
   give_me_some_credit_{dataset}_raw      give_me_some_credit
              │                                      │
              ▼                                      ▼
   01_eda_bronze.py                        02_eda_silver.py
   "Como o dado chegou?"                   "O tratamento funcionou?"
              │                                      │
              └───────────────┬──────────────────────┘
                              ▼
                 src/quality/profiling.py
        (schema, nulos, duplicidade, cardinalidade,
         estatísticas, amostra)
```

### A regra de ouro: EAD só diagnostica

| Fase | Papel | Altera o dado? |
|---|---|---|
| **EAD** | Investigar e reportar problemas | **Não** |
| **Silver** | Aplicar os tratamentos decididos a partir da EAD | **Sim** |

Essa separação foi confirmada no histórico do projeto: a EAD é puramente investigativa, e é na **construção da Silver** que as decisões viram código.

### Os dois papéis da EAD neste projeto

1. **Antes do tratamento (Bronze):** descobrir o que há de errado nos dados brutos (nulos, códigos de erro, duplicidade). Isso **alimenta** as regras da Silver.
2. **Depois do tratamento (Silver):** conferir se as regras funcionaram e quantos registros caíram em "quarentena". Isso **audita** a Silver.

---

## 2. Como o código está organizado

O padrão é o mesmo do resto do projeto: **a lógica fica em `src/`; os notebooks só orquestram.**

| Onde | O que contém |
|---|---|
| `notebooks/ead/01_eda_bronze.py` | Orquestra o diagnóstico da **Bronze** |
| `notebooks/ead/02_eda_silver.py` | Orquestra o diagnóstico da **Silver** e adiciona gráficos e checagens |
| `src/quality/profiling.py` | **Toda a lógica** de diagnóstico, reutilizável para Bronze, Silver e uma futura Gold |

### Por que extrair o diagnóstico para `profiling.py`?

Na versão original, cada notebook reescrevia a mesma lógica (nulos, duplicidade, cardinalidade). Agora ela vive **em um único lugar**: para perfilar uma nova tabela, basta chamar `run_profile`.

### O que a versão original fazia de errado

O `EAD_silver.ipynb` original **recalculava toda a limpeza Bronze → Silver dentro do próprio notebook de EAD**. Isso duplicava a regra de negócio do `silver.py`: se a regra mudasse lá, a EAD continuaria mostrando o comportamento antigo, sem avisar. Hoje o notebook faz só `spark.table(...)`, **lendo a tabela real** e nunca recalculando.

---

## 3. O módulo `profiling.py`

São cinco funções. A `run_profile` chama as outras em sequência.

| Função | O que faz | Devolve |
|---|---|---|
| `null_completeness_report` | Conta nulos e o percentual por coluna, do pior para o melhor | DataFrame (`coluna`, `tipo_dado`, `qtd_nulos_ou_vazios`, `pct_nulos`) |
| `duplicate_report` | Verifica linhas 100% idênticas e chaves de negócio repetidas | Só imprime |
| `cardinality_report` | Conta valores distintos (aproximado) por coluna | DataFrame (`coluna`, `valores_distintos_aprox`, `pct_distintos`) |
| `numeric_summary` | Estatísticas das colunas numéricas: contagem, média, desvio, mínimo, 25%, 50%, 75%, máximo | DataFrame |
| `run_profile` | Orquestra tudo e imprime o relatório completo | O DataFrame de nulos |

### O que o `run_profile` imprime (nesta ordem)

```
1. Cabeçalho ........ total de linhas e de colunas da tabela
2. SCHEMA ........... nomes e tipos de cada coluna
3. COMPLETUDE ....... nulos por coluna (qtd e %)
4. DUPLICIDADE ...... linhas idênticas e chaves repetidas   (se key_cols foi informado)
5. CARDINALIDADE .... valores distintos por coluna
6. NUMÉRICO ......... mínimo, percentis e máximo            (se numeric_cols foi informado)
7. AMOSTRA .......... 5 primeiras linhas
```

### Parâmetros do `run_profile`

| Parâmetro | Função |
|---|---|
| `df` | O DataFrame a analisar |
| `table_label` | Nome exibido no cabeçalho (o nome da tabela) |
| `key_cols` | Chave de negócio para checar duplicidade (aqui, `["customer_id"]`) |
| `numeric_cols` | Colunas para o sumário estatístico. Colunas que não existirem são ignoradas |

---

## 4. Notebook 1 — EAD da Bronze (`01_eda_bronze.py`)

### 4.1 Passo a passo

1. **Widgets:** `catalog` (`credito_dev`, `credito_hml` ou `credito_prd`) e `dataset` (`training` ou `scoring`).
2. **Importa** `run_profile` do pacote `src` (procurando a raiz do repositório).
3. **Monta a tabela:** `{catalog}.bronze.give_me_some_credit_{dataset}_raw`.
4. **Lista as colunas numéricas** com os **nomes originais do Kaggle** (`SeriousDlqin2yrs`, `RevolvingUtilizationOfUnsecuredLines`, `age`, `DebtRatio`, `MonthlyIncome` etc.), pois a Bronze ainda não foi renomeada.
5. **Roda o diagnóstico** com `key_cols=["customer_id"]`.
6. **Exibe** o relatório de nulos com `display()`.

### 4.1.1 Por que existe o widget `dataset`?

Porque a Bronze tem **uma tabela por dataset** (`training` e `scoring`). Antes, o notebook lia uma tabela única e contaminada. Agora cada execução analisa **um dataset por vez**.

### 4.2 O que procurar na saída da Bronze

| Bloco do relatório | O que observar | Por que importa |
|---|---|---|
| **Total de linhas** | Bate com o esperado do arquivo de origem? | Detecta mistura de arquivos |
| **Completude** | Quais colunas têm nulos e em que proporção | Define o tratamento de nulos na Silver |
| **Duplicidade por `customer_id`** | Deve ser zero | Chaves repetidas indicam reingestão ou problema na origem |
| **Cardinalidade** | Colunas com poucos valores distintos (candidatas a categóricas ou contagens) e `customer_id` com ~100% distinto | Confirma quem é chave e quem é variável |
| **Sumário numérico** | Máximos e mínimos estranhos (idade 0, atrasos em 96/98, razões muito altas) | Revela **valores impossíveis** e códigos de erro |

### 4.3 O que já se sabe sobre esse dataset

Estes pontos vêm do conhecimento público do dataset do Kaggle, **não da sua execução**. Confirme cada um na saída do seu ambiente:

| Característica | O que costuma aparecer | Tratamento na Silver |
|---|---|---|
| `MonthlyIncome` com muitos nulos (perto de 20%) | Renda ausente | Flag `is_monthly_income_null` |
| `NumberOfDependents` com poucos nulos (perto de 3%) | Dependentes ausentes | Nulo vira `0` |
| `age` com valores fora da faixa plausível | Idade zero ou muito alta | Fora de 18–120 vira `null` |
| Colunas de atraso com valores 96 e 98 | Códigos de erro, não contagens reais | Flag `has_delinquency_outlier` |
| Alvo desbalanceado (poucos inadimplentes) | Classe 1 muito menor que a classe 0 | Tratado na modelagem, não na Silver |

---

## 5. Notebook 2 — EAD da Silver (`02_eda_silver.py`)

### 5.1 Passo a passo

1. **Widget:** apenas `catalog`. Este notebook lê **somente a Silver de treino** (`silver.give_me_some_credit`), não a de scoring.
2. **Roda o diagnóstico** (`run_profile`) usando os **nomes já padronizados** (`debt_ratio`, `monthly_income`, `revolving_utilization_unsecured`, `num_times_30_59_days_late` etc.).
3. **Distribuição do alvo** (`target_default_2yrs`): contagem e percentual de adimplentes (0) e inadimplentes (1).
4. **Gráfico de barras** da distribuição do alvo, via matplotlib (versão estática, com rótulos nas barras).
5. **Checagem de qualidade** com as flags herdadas da Silver.

### 5.2 A distribuição do alvo

| Por que olhar | O que revela |
|---|---|
| Proporção 0 × 1 | O **desbalanceamento** da base: como poucos clientes são inadimplentes, a acurácia sozinha engana |
| Impacto | Justifica métricas como ROC-AUC e PR-AUC, e a escolha do threshold (ver documento de hipóteses) |

O notebook mostra o mesmo gráfico de duas formas: a tabela nativa do Databricks (`display`) e a versão estática em matplotlib, para quem preferir uma imagem fixa.

### 5.3 A checagem de qualidade (a parte mais importante da EAD Silver)

O último bloco soma as flags criadas pela Silver e conta as idades nulas:

| Indicador | Origem | O que significa |
|---|---|---|
| `qtd_renda_nula` | Soma de `is_monthly_income_null` | Quantos clientes não têm renda informada |
| `qtd_outlier_atraso` | Soma de `has_delinquency_outlier` | Quantos têm código de erro (≥ 96) em alguma coluna de atraso |
| `qtd_idade_em_quarentena` | Idades `null` | Quantos tiveram a idade invalidada (fora de 18–120) |

Isso é a **quarentena lógica** da Silver em números: em vez de descartar registros, a Silver os marca. A EAD mostra **quantos** foram marcados, o que permite avaliar se o tratamento foi brando ou agressivo demais.

---

## 6. Como a EAD se conecta à Silver

A EAD encontra o problema; a Silver aplica o tratamento. Veja o "de-para" completo:

| Problema encontrado | Onde aparece na EAD | Tratamento na Silver (`silver.py`) |
|---|---|---|
| Renda mensal nula | Completude; flag na EAD Silver | Cria `is_monthly_income_null`; o valor continua nulo (não é imputado) |
| Dependentes nulos | Completude | Nulo vira `0` |
| Idade fora da faixa | Sumário numérico (mínimo e máximo) | Fora de 18–120 vira `null` |
| Atrasos com 96 ou 98 | Sumário numérico (máximo) | Cria `has_delinquency_outlier`; o valor original é mantido |
| `customer_id` nulo | Completude | Linha removida |
| Chaves repetidas | Duplicidade | Deduplicação por `customer_id` e MERGE |

> **Um detalhe didático:** a Silver **não imputa** a renda. Ela apenas sinaliza. A decisão de como lidar com esse nulo (imputar, manter para o XGBoost lidar, criar categoria) fica para a modelagem, e a flag garante que a informação "a renda estava ausente" não se perca.

---

## 7. Análises complementares discutidas no histórico

Além dos dois notebooks anexados, o histórico do projeto descreve outras análises típicas de EAD para um problema de crédito. Confirme se estão no seu repositório:

| Análise | Para que serve |
|---|---|
| **Distribuição do alvo** | Medir o desbalanceamento (já coberta na EAD Silver) |
| **Distribuições e correlações das variáveis** | Ver formato de cada variável e relações entre elas |
| **WOE e IV** | Medidas clássicas de crédito: mostram o poder de cada variável para separar bons e maus pagadores |
| **Mecanismo dos nulos: MCAR, MAR ou MNAR** | Entender **por que** faltam dados, o que muda o tratamento |
| **Relatório de EAD para o negócio** | Resumo curto e objetivo para quem decide |

### Os três tipos de dado ausente (didático)

| Tipo | Significado | Exemplo no projeto |
|---|---|---|
| **MCAR** | Falta ao acaso, sem relação com nada | Erro aleatório de digitação |
| **MAR** | A falta depende de **outra variável observada** | Renda ausente mais comum em certa faixa de idade |
| **MNAR** | A falta depende do **próprio valor** ausente ou do alvo | Clientes de maior risco deixam de informar a renda |

O histórico descreve um notebook de investigação de valores ausentes (`02_analise_missing_values.py`) com quatro etapas: mapear a taxa de nulos por coluna, correlacionar os padrões de nulidade, testar MAR e testar MNAR por associação (qui-quadrado) entre a nulidade e o alvo, terminando em uma **tabela de decisões de qualidade** (um registro do tratamento escolhido para cada coluna).

---

## 8. Ordem de execução

```
00_reset_ambiente   (uma vez)
01_bronze           (dataset = training  e  dataset = scoring)
02_silver           (idem)
03_gold
   │
   ├─ notebooks/ead/01_eda_bronze.py   (escolher training ou scoring)
   └─ notebooks/ead/02_eda_silver.py
```

Os notebooks de EAD **precisam** que a camada correspondente já exista, pois só leem tabelas prontas.

> **Observação sobre a ordem:** no fluxo do repositório, a EAD roda **depois** das tabelas prontas. Por isso, aqui ela funciona mais como **auditoria** do que como investigação prévia. Em um projeto novo, a investigação (nulos, códigos de erro, distribuições) viria **antes** de escrever as regras da Silver.

---

## 9. Problemas já encontrados e como foram resolvidos

| Sintoma | Causa | Solução |
|---|---|---|
| EAD da Bronze mostrava **251.503 linhas** | Tabela contaminada: treino e teste lidos juntos por um glob genérico | Bronze separada por dataset; widget `dataset` no notebook |
| EAD Silver recalculava a limpeza | Lógica de `silver.py` duplicada dentro do notebook | Notebook passou a só ler a tabela Silver |
| Três rascunhos quase idênticos do gráfico do alvo | Versões intermediárias acumuladas | Ficaram só a versão nativa (`display`) e a final em matplotlib |
| Diagnóstico repetido em cada notebook | Lógica escrita direto em cada um | Extraída para `src/quality/profiling.py` |

---

## 10. Pontos de atenção (para revisar)

1. **`%uv sync` ativo nos dois notebooks.** Nos arquivos anexados, a linha `%uv sync` está solta na linha 7, fora de comentário. Segundo o histórico, é exatamente esse padrão que causou `SyntaxError` no lint de CI (e que o `check_notebook_hygiene.py` bloqueia). Sugestão: comentar (`# %uv sync`) ou usar o padrão adotado nos demais notebooks.
2. **O título "Visualização gráfica da completude" promete mais do que o código entrega.** O `display(df_diagnostico_nulos)` mostra uma **tabela**. O gráfico só aparece se você o configurar na interface do Databricks (aba de visualização).
3. **A EAD Silver não cobre o `scoring`.** Ela lê só `silver.give_me_some_credit`. A Silver de scoring (`give_me_some_credit_scoring`) não tem diagnóstico próprio, embora seja ela que alimenta a inferência.
4. **A checagem de duplicidade por linha inteira pode falhar na Bronze.** Como a Bronze tem `_ingestion_timestamp`, uma mesma linha reingerida em outra execução deixa de ser "100% idêntica". A checagem **por chave** (`customer_id`) é a que pega esse caso.
5. **O nome da coluna `qtd_nulos_ou_vazios` engana um pouco.** O código conta apenas nulos verdadeiros. Como as colunas são numéricas, isso basta, mas texto vazio não seria contado.
6. **A EAD ainda não valida o "alvo nulo" da Gold.** É a Gold que junta treino e scoring e cria as linhas com `target_dlq_2yrs` nulo por construção. Um diagnóstico da Gold (`03_eda_gold.py`, citado no histórico como possível extensão) fecharia esse ponto.

---

## 11. Glossário rápido

| Termo | Significado simples |
|---|---|
| **EAD / EDA** | Análise exploratória: olhar os dados sem alterá-los |
| **Profiling** | Relatório automático de qualidade do dado (nulos, duplicidade, estatísticas) |
| **Completude** | Quanto de cada coluna está preenchido |
| **Cardinalidade** | Quantidade de valores distintos de uma coluna |
| **Percentis (25%, 50%, 75%)** | Pontos que dividem os dados ordenados; 50% é a mediana |
| **Outlier / código de erro** | Valor fora do esperado (aqui, 96 e 98 nas colunas de atraso) |
| **Quarentena lógica** | Marcar ou anular o dado inválido em vez de apagar a linha |
| **Desbalanceamento** | Uma classe do alvo muito menos frequente que a outra |
| **WOE / IV** | Medidas de crédito para o poder de uma variável separar bons e maus pagadores |
| **MCAR / MAR / MNAR** | Três causas possíveis para dados ausentes |
| **Chave de negócio** | Coluna que identifica o registro (aqui, `customer_id`) |
