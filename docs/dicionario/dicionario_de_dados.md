# Dicionário de dados — risco_credito_databricks

> Gerado automaticamente a partir de `src/config/data_dictionary.py`. **Não editar à mão** — altere o módulo e rode `python -m src.config.data_dictionary`.
> Os mesmos textos são aplicados como `COMMENT` no Unity Catalog (`apply_all`), que é a fonte consultável no Catalog Explorer.

## Regras de limpeza (Silver)

- Deduplicação por `customer_id`.
- `age` fora do intervalo 18–115 é convertida em nulo.
- `num_dependents` nulo é convertido em 0.
- `is_monthly_income_null` = 1 quando `monthly_income` é nulo (a renda NÃO é imputada).
- `has_delinquency_outlier` = 1 quando qualquer coluna de atraso (30-59, 60-89, 90+) é maior ou igual a 96.
- Nenhuma linha é descartada: anomalias viram flag (quarentena lógica).

## `<catalog>.silver.give_me_some_credit`

Silver do dataset Give Me Some Credit (treino): dados limpos, padronizados e deduplicados por customer_id, com flags de qualidade.

Tags: `camada=silver`, `projeto=risco_credito_databricks`

| Coluna | Descrição | Origem |
|---|---|---|
| `customer_id` | Identificador do cliente (chave de negócio; a tabela é deduplicada por esta coluna). | Kaggle: customer_id |
| `target_default_2yrs` | Alvo: 1 = atraso de 90 dias ou mais nos últimos 2 anos; 0 = sem atraso. Nulo/ausente no dataset de scoring. | Kaggle: SeriousDlqin2yrs |
| `revolving_utilization_unsecured` | Saldo total em cartões de crédito e linhas pessoais (exceto imóveis e parcelados) dividido pela soma dos limites de crédito. | Kaggle: RevolvingUtilizationOfUnsecuredLines |
| `age` | Idade do cliente em anos. Valores fora de 18–115 são anulados. | Kaggle: age |
| `num_times_30_59_days_late` | Número de vezes com atraso de 30 a 59 dias nos últimos 2 anos. | Kaggle: NumberOfTime30-59DaysPastDueNotWorse |
| `debt_ratio` | Pagamentos mensais de dívidas, pensão e custos de vida divididos pela renda bruta mensal. | Kaggle: DebtRatio |
| `monthly_income` | Renda mensal do cliente. Pode ser nula (ver is_monthly_income_null). | Kaggle: MonthlyIncome |
| `num_open_credit_lines_and_loans` | Número de empréstimos abertos (ex.: financiamento de carro) e linhas de crédito (ex.: cartões). | Kaggle: NumberOfOpenCreditLinesAndLoans |
| `num_times_90_days_late` | Número de vezes com atraso de 90 dias ou mais. | Kaggle: NumberOfTimes90DaysLate |
| `num_real_estate_loans_or_lines` | Número de financiamentos imobiliários e linhas de crédito com garantia de imóvel. | Kaggle: NumberRealEstateLoansOrLines |
| `num_times_60_89_days_late` | Número de vezes com atraso de 60 a 89 dias nos últimos 2 anos. | Kaggle: NumberOfTime60-89DaysPastDueNotWorse |
| `num_dependents` | Número de dependentes do cliente (exceto ele próprio). Nulo vira 0 no Silver. | Kaggle: NumberOfDependents |
| `is_monthly_income_null` | Flag de qualidade: 1 = renda mensal nula na origem; 0 = informada. Não é propagada para o Gold. | Derivada (Silver) |
| `has_delinquency_outlier` | Flag de qualidade: 1 = alguma coluna de atraso com valor >= 96 (valor atípico da origem); 0 = sem outlier. Não é propagada para o Gold. | Derivada (Silver) |
| `_ingestion_timestamp` | Momento em que a linha foi ingerida no Bronze (linhagem). | Metadado de pipeline |
| `_source_file` | Arquivo de origem da linha (linhagem). | Metadado de pipeline |
| `_silver_processed_at` | Momento em que a linha foi processada para o Silver (linhagem). | Metadado de pipeline |

## `<catalog>.silver.give_me_some_credit_scoring`

Silver do dataset Give Me Some Credit (scoring): lote sem rótulo a ser pontuado pelo modelo. Mesmo schema do treino.

Tags: `camada=silver`, `projeto=risco_credito_databricks`

| Coluna | Descrição | Origem |
|---|---|---|
| `customer_id` | Identificador do cliente (chave de negócio; a tabela é deduplicada por esta coluna). | Kaggle: customer_id |
| `target_default_2yrs` | Alvo: 1 = atraso de 90 dias ou mais nos últimos 2 anos; 0 = sem atraso. Nulo/ausente no dataset de scoring. | Kaggle: SeriousDlqin2yrs |
| `revolving_utilization_unsecured` | Saldo total em cartões de crédito e linhas pessoais (exceto imóveis e parcelados) dividido pela soma dos limites de crédito. | Kaggle: RevolvingUtilizationOfUnsecuredLines |
| `age` | Idade do cliente em anos. Valores fora de 18–115 são anulados. | Kaggle: age |
| `num_times_30_59_days_late` | Número de vezes com atraso de 30 a 59 dias nos últimos 2 anos. | Kaggle: NumberOfTime30-59DaysPastDueNotWorse |
| `debt_ratio` | Pagamentos mensais de dívidas, pensão e custos de vida divididos pela renda bruta mensal. | Kaggle: DebtRatio |
| `monthly_income` | Renda mensal do cliente. Pode ser nula (ver is_monthly_income_null). | Kaggle: MonthlyIncome |
| `num_open_credit_lines_and_loans` | Número de empréstimos abertos (ex.: financiamento de carro) e linhas de crédito (ex.: cartões). | Kaggle: NumberOfOpenCreditLinesAndLoans |
| `num_times_90_days_late` | Número de vezes com atraso de 90 dias ou mais. | Kaggle: NumberOfTimes90DaysLate |
| `num_real_estate_loans_or_lines` | Número de financiamentos imobiliários e linhas de crédito com garantia de imóvel. | Kaggle: NumberRealEstateLoansOrLines |
| `num_times_60_89_days_late` | Número de vezes com atraso de 60 a 89 dias nos últimos 2 anos. | Kaggle: NumberOfTime60-89DaysPastDueNotWorse |
| `num_dependents` | Número de dependentes do cliente (exceto ele próprio). Nulo vira 0 no Silver. | Kaggle: NumberOfDependents |
| `is_monthly_income_null` | Flag de qualidade: 1 = renda mensal nula na origem; 0 = informada. Não é propagada para o Gold. | Derivada (Silver) |
| `has_delinquency_outlier` | Flag de qualidade: 1 = alguma coluna de atraso com valor >= 96 (valor atípico da origem); 0 = sem outlier. Não é propagada para o Gold. | Derivada (Silver) |
| `_ingestion_timestamp` | Momento em que a linha foi ingerida no Bronze (linhagem). | Metadado de pipeline |
| `_source_file` | Arquivo de origem da linha (linhagem). | Metadado de pipeline |
| `_silver_processed_at` | Momento em que a linha foi processada para o Silver (linhagem). | Metadado de pipeline |

## `<catalog>.gold.dim_customer`

Dimensão de cliente: atributos demográficos estáveis por customer_id.

Tags: `camada=gold`, `projeto=risco_credito_databricks`

| Coluna | Descrição | Origem |
|---|---|---|
| `customer_id` | Identificador do cliente (chave de negócio; a tabela é deduplicada por esta coluna). | Silver: customer_id |
| `age` | Idade do cliente em anos. Valores fora de 18–115 são anulados. | Silver: age |
| `num_dependents` | Número de dependentes do cliente (exceto ele próprio). Nulo vira 0 no Silver. | Silver: num_dependents |
| `_gold_processed_at` | Momento em que a linha foi processada para o Gold (linhagem). | Metadado de pipeline |

## `<catalog>.gold.fct_credit_profile`

Fato de perfil de crédito: variáveis explicativas do modelo, feature derivada total_delinquency_events e o alvo. Inclui as linhas de scoring (alvo nulo) quando o Silver de scoring existe.

Tags: `camada=gold`, `projeto=risco_credito_databricks`

| Coluna | Descrição | Origem |
|---|---|---|
| `customer_id` | Identificador do cliente (chave de negócio; a tabela é deduplicada por esta coluna). | Silver: customer_id |
| `monthly_income` | Renda mensal do cliente. Pode ser nula (ver is_monthly_income_null). | Silver: monthly_income |
| `debt_ratio` | Pagamentos mensais de dívidas, pensão e custos de vida divididos pela renda bruta mensal. | Silver: debt_ratio |
| `revolving_utilization` | Saldo total em cartões de crédito e linhas pessoais (exceto imóveis e parcelados) dividido pela soma dos limites de crédito. | Silver: revolving_utilization_unsecured |
| `num_open_credit_lines` | Número de empréstimos abertos (ex.: financiamento de carro) e linhas de crédito (ex.: cartões). | Silver: num_open_credit_lines_and_loans |
| `num_real_estate_loans` | Número de financiamentos imobiliários e linhas de crédito com garantia de imóvel. | Silver: num_real_estate_loans_or_lines |
| `num_times_30_59_days_late` | Número de vezes com atraso de 30 a 59 dias nos últimos 2 anos. | Silver: num_times_30_59_days_late |
| `num_times_60_89_days_late` | Número de vezes com atraso de 60 a 89 dias nos últimos 2 anos. | Silver: num_times_60_89_days_late |
| `num_times_90_days_late` | Número de vezes com atraso de 90 dias ou mais. | Silver: num_times_90_days_late |
| `total_delinquency_events` | Feature derivada: soma dos atrasos de 30-59, 60-89 e 90+ dias. | Derivada (Gold): num_times_30_59_days_late + num_times_60_89_days_late + num_times_90_days_late |
| `target_dlq_2yrs` | Alvo: 1 = atraso de 90 dias ou mais nos últimos 2 anos; 0 = sem atraso. Nulo/ausente no dataset de scoring. | Silver: target_default_2yrs |
