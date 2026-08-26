# Risco de Crédito — Score de Probabilidade de Inadimplência  ok

*Parte da série **Databricks de Ponta a Ponta***

![CI](https://github.com/<seu-usuario>/risco-credito-databricks/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Problema de negócio
Uma instituição financeira precisa decidir, para cada solicitação de crédito, se aprova ou não a operação — e com qual taxa de juros. A decisão precisa ser rápida, consistente entre analistas, e defensável em auditoria. O objetivo deste projeto é construir um score de probabilidade de inadimplência que sustente essa decisão de ponta a ponta: da ingestão do dado até um endpoint em produção monitorado.

## Base de dados
- **Give Me Some Credit** (Kaggle) — https://www.kaggle.com/c/GiveMeSomeCredit — ~150.000 registros de clientes pessoa física, 10 variáveis explicativas (utilização de crédito rotativo, idade, número de dependentes, histórico de atraso, dívida/renda, entre outras) e variável-alvo `SeriousDlqin2yrs` (1 = inadimplência em 90+ dias nos próximos 2 anos)
- **SCR.data** (BACEN, dados agregados) — https://dadosabertos.bcb.gov.br/dataset/scr_data — usado como referência externa para comparar a taxa de inadimplência do modelo com a média nacional por segmento, não como dado de treino

## Métrica-alvo
- **Métrica primária:** AUC-ROC ≥ 0,85 no conjunto de teste (referência de mercado: soluções de topo da competição original Kaggle atingem ~0,86–0,87)
- **Métricas complementares:** KS-statistic (separação entre bons e maus pagadores), recall na classe inadimplente a um nível de precisão definido pelo negócio
- **Métrica de negócio:** estimativa de redução de inadimplência em R$ comparando a política "com modelo" vs. a política atual (aprovação sem score)

## Hipóteses
1. **H1 — Utilização de crédito rotativo:** clientes com maior `RevolvingUtilizationOfUnsecuredLines` (percentual do limite de crédito já utilizado) têm probabilidade significativamente maior de inadimplência nos próximos 2 anos, porque um uso próximo do limite indica menor folga financeira.
2. **H2 — Dívida/renda como preditor dominante:** o `DebtRatio` (relação dívida/renda) tem poder preditivo maior que a idade do cliente isoladamente — ou seja, a capacidade de pagamento pesa mais na inadimplência do que o perfil demográfico.

## Perguntas de investigação
1. **Quais variáveis têm maior poder preditivo** para inadimplência, segundo a importância de features via SHAP — e essa ordem confirma ou contraria as hipóteses H1 e H2?
2. **Existe um ponto de corte (threshold) de probabilidade** que equilibra de forma ótima aprovar bons pagadores e reter maus pagadores, considerando o custo assimétrico entre os dois tipos de erro (negar crédito bom vs. aprovar crédito ruim)?
3. **O modelo mantém performance estável entre diferentes faixas de renda e idade**, ou há uma faixa em que a taxa de erro é desproporcionalmente maior (checagem de viés antes de colocar em produção)?
4. **Qual o impacto financeiro estimado** (redução de inadimplência em R$, usando um valor médio de operação de crédito como referência) se o modelo fosse adotado em produção, comparado à política atual sem modelo?

## Modelo
- **Modelo principal:** Gradient Boosting (XGBoost) — dado tabular, alvo binário desbalanceado (~6,7% de positivos), e a necessidade de capturar interações não lineares entre variáveis financeiras torna Gradient Boosting a escolha natural; é também o tipo de modelo que consistentemente performa melhor nas soluções de referência dessa competição
- **Modelo de comparação (baseline interpretável):** Regressão Logística — usada como benchmark de interpretabilidade direta (coeficientes) e como piso de performance; a decisão final de qual modelo vai para produção é documentada comparando os dois, não assumida de antemão
- **Explicabilidade:** SHAP obrigatório para o modelo principal — cada decisão de aprovação/negação precisa ser rastreável até as variáveis que mais pesaram, para sustentar a resposta ao cliente e à auditoria
- **Tratamento do desbalanceamento:** class weighting ou SMOTE, com decisão documentada de qual técnica foi usada e por quê

## Arquitetura no Databricks (pipeline ponta a ponta)
1. **Bronze** — ingestão via Autoloader do CSV bruto
2. **Silver** — tratamento de outliers (ex.: idade = 0, utilização de crédito > 10), imputação de nulos, padronização de schema
3. **Gold / Feature Store** — feature table versionada, pronta para treino
4. **Treino** — MLflow tracking de experimentos, comparação XGBoost vs. Regressão Logística
5. **Registro** — MLflow Model Registry, promoção do modelo vencedor
6. **Produção** — Model Serving, endpoint real
7. **Monitoramento** — Lakehouse Monitoring: drift de dados de entrada e de performance, alerta automático por limiar
8. **Governança** — Unity Catalog com RBAC controlando quem acessa a feature table e o endpoint

## Limitações assumidas
- O dado é de 2011 (competição original) e de clientes americanos — usado aqui como base metodológica, não como fonte de verdade sobre o mercado brasileiro; o cruzamento com o SCR.data serve para contextualizar essa diferença, não para corrigi-la
- A variável-alvo tem 24 meses de horizonte; decisões de crédito com prazo diferente exigiriam reavaliação do problema

## Status
Em andamento — ingestão Bronze já implementada (`01_ingestao_bronze_autoloader.py`), seguindo roadmap de 9 fases.
