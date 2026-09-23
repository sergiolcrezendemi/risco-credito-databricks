# Risco de Crédito — Score de Probabilidade de Inadimplência

*Parte da série **Databricks de Ponta a Ponta***

![CI](https://github.com/<seu-usuario>/risco-credito-databricks/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

> Projeto de estudo com dados públicos (Kaggle *Give Me Some Credit*). Valores financeiros são ilustrativos e não representam dados de nenhuma instituição real.

## Resumo executivo

| Indicador | Resultado |
|---|---|
| ROC-AUC do modelo principal (XGBoost) | **0,867** (meta ≥ 0,85) — regressão logística: 0,788 |
| Régua de corte escolhida (threshold) | **0,45** |
| Taxa de aprovação | 100% → **72,8%** |
| Inadimplência da carteira aprovada | 6,68% → **1,75%** (queda de ~74%) |
| Impacto financeiro líquido estimado | **+R$ 14,5 mi** (parâmetros ilustrativos) |
| Monitoramento pós-deploy | Data drift (PSI + KS por feature) e concept drift (queda de AUC-ROC) com histórico em Delta e MLflow |

**Em uma frase:** o risco é explicado principalmente pelo **comportamento** do cliente (uso do limite e histórico de atrasos), e um corte em 0,45 reduz a inadimplência em ~74% mantendo a aprovação acima do piso de 70%. O peso da idade no modelo exige checagem de viés antes de qualquer uso real.

## Problema de negócio
Uma instituição financeira precisa decidir, para cada solicitação de crédito, se aprova ou não a operação — e com qual taxa de juros. A decisão precisa ser rápida, consistente entre analistas e defensável em auditoria. O objetivo deste projeto é construir um score de probabilidade de inadimplência que sustente essa decisão de ponta a ponta: da ingestão do dado até um endpoint em produção monitorado.

## Base de dados
- **Give Me Some Credit** (Kaggle) — https://www.kaggle.com/c/GiveMeSomeCredit — ~150.000 registros de clientes pessoa física, 10 variáveis explicativas (utilização de crédito rotativo, idade, número de dependentes, histórico de atraso, dívida/renda, entre outras) e variável-alvo `SeriousDlqin2yrs` (1 = inadimplência em 90+ dias nos próximos 2 anos)
- **SCR.data** (BACEN, dados agregados) — https://dadosabertos.bcb.gov.br/dataset/scr_data — usado como referência externa para comparar a taxa de inadimplência do modelo com a média nacional por segmento, não como dado de treino

## Métrica-alvo
- **Métrica primária:** AUC-ROC ≥ 0,85 no conjunto de teste (referência de mercado: soluções de topo da competição original Kaggle atingem ~0,86–0,87) — **atingida: 0,867**
- **Métricas complementares:** KS-statistic (separação entre bons e maus pagadores), recall na classe inadimplente a um nível de precisão definido pelo negócio
- **Métrica de negócio:** estimativa de redução de inadimplência em R$ comparando a política "com modelo" vs. a política atual (aprovação sem score)

## Hipóteses

As quatro hipóteses seguem uma progressão em duas camadas. Não faz sentido decidir threshold ou calcular impacto em R$ (Camada 2) de um modelo em que ainda não se confia (Camada 1).

- **Camada 1 — Confiança no modelo (H1 e H2):** o modelo aprende algo que faz sentido de negócio, ou acerta por motivos que ninguém consegue explicar?
- **Camada 2 — Decisão operacional (H3 e H4):** dado que o modelo é confiável, que corte usar e se vale a pena financeiramente.

| Hipótese | Enunciado | Como foi testada | Decisão que sustenta |
|---|---|---|---|
| **H1** — Utilização do rotativo | Clientes com maior `revolving_utilization` têm probabilidade significativamente maior de inadimplência, porque uso próximo do limite indica menor folga financeira | Relação entre o valor da feature e seu próprio SHAP value | Confiar ou não no modelo |
| **H2** — Dívida/renda como preditor dominante | `debt_ratio` tem poder preditivo maior que `age` — capacidade de pagamento pesa mais que perfil demográfico | Comparação do \|SHAP\| médio entre `debt_ratio` e `age`, com margem relativa mínima de 10% | Aprovar o modelo em compliance/auditoria |
| **H3** — Threshold ótimo | Existe um ponto de corte que reduz a inadimplência da carteira aprovada sem derrubar a aprovação abaixo de um piso viável | Varredura de thresholds sobre `y_pred_xgb`, filtrando pelo piso de aprovação (70%, ilustrativo) e escolhendo o de menor inadimplência entre os viáveis | Definir a regra operacional de aprovação |
| **H4** — Impacto financeiro | O modelo teria evitado mais perdas em R$ do que o custo de rejeitar bons pagadores | Perda evitada (maus rejeitados × custo médio de inadimplência) vs. custo de oportunidade (bons rejeitados × margem média), no threshold de H3 | Aprovar (ou não) o investimento em produção |

## Resultados das hipóteses

Os gráficos abaixo foram gerados no notebook `01_hipoteses_h1_h4`.

### Importância das variáveis (ranking SHAP)

![Ranking SHAP](imagens/00_ranking_shap.png)

Duas variáveis dominam com folga: **uso do crédito rotativo** (|SHAP| médio 0,75) e **total de eventos de atraso** (0,68). Depois vem um degrau grande: idade (0,25), número de linhas de crédito abertas (0,20) e dívida/renda (0,15).

**Leitura:** o risco é explicado principalmente pelo comportamento (como o cliente usa o crédito e se já atrasou), e não pelo perfil (renda, dependentes, imóveis).

> Os atrasos de 30–59, 60–89 e 90+ dias aparecem no fim do ranking provavelmente porque já estão somados em `total_delinquency_events`. A informação não é irrelevante: está concentrada na variável agregada.

### H1 — Uso do rotativo vs. impacto no risco ✅ Confirmada

![H1](imagens/h1_rotativo_shap.png)

Cada ponto é um cliente: eixo horizontal = fração do limite utilizada; eixo vertical = contribuição SHAP para o risco. A relação é praticamente uma escada subindo: uso perto de 0% reduz o risco, por volta de **35–40%** a variável passa a prejudicar e perto de **100% (limite estourado)** o risco dispara.

**Implicação:** o uso do limite é o alerta mais forte do modelo e pode virar regra operacional — monitorar clientes da carteira que cruzam ~40% e ~90% de uso.

> **Nota metodológica:** a correlação de Pearson exibida no gráfico (0,071) subestima a relação, porque é distorcida por outliers extremos (há clientes com utilização na casa dos milhares, cortados pelo `xlim` em 2,0). A correlação de Spearman, ou o cálculo restrito ao intervalo 0–1, representa melhor a relação monotônica visível no gráfico.

### H2 — Dívida/renda vs. idade ❌ Contrariada

![H2](imagens/h2_debt_ratio_vs_age.png)

A hipótese era que dívida/renda superaria idade. Deu o contrário: idade (|SHAP| 0,246) pesa **41% a mais** que dívida/renda (0,145).

**Implicações:**
1. **Negócio:** vale mais olhar *como* o cliente usa o crédito (H1) do que o índice de endividamento declarado — renda autodeclarada costuma ser pouco confiável.
2. **Risco regulatório:** o peso da idade pode gerar tratamento desigual por faixa etária. Isso será verificado no notebook 02 (viés por idade/renda) antes de qualquer uso real.

### H3 — Aprovação vs. inadimplência por threshold ✅ Confirmada

![H3](imagens/h3_threshold.png)

Linha azul: % de aprovados (eixo esquerdo). Linha vermelha: % de inadimplentes entre os aprovados (eixo direito). Tracejado cinza: piso de aprovação (70%). Pontilhado vermelho: inadimplência sem modelo (6,68%). Linha verde: régua escolhida.

| | Sem modelo | Com modelo (threshold 0,45) |
|---|---|---|
| Taxa de aprovação | 100% | 72,8% |
| Inadimplência da carteira | 6,68% | **1,75%** |

**Implicação:** recusando ~27% dos pedidos, a inadimplência cai ~74%. O threshold funciona como botão de estratégia: subir a régua favorece crescimento com mais risco; descer protege caixa. O gráfico mostra o preço de cada escolha.

### H4 — Impacto financeiro líquido ✅ Confirmada

![H4 impacto financeiro](imagens/h4_impacto_financeiro.png)

| Componente | Cálculo | Valor |
|---|---|---|
| Perda evitada | 1.622 maus pagadores recusados × ~R$ 15 mil | +R$ 24,3 mi |
| Custo de oportunidade | 6.538 bons pagadores recusados × ~R$ 1,5 mil | –R$ 9,8 mi |
| **Impacto líquido** | | **+R$ 14,5 mi** |

**Implicação:** de cada 5 clientes recusados, só 1 era de fato mau pagador. Mesmo assim o modelo compensa, porque um calote custa ~10 vezes a margem de um bom cliente.

> **Ressalva:** perda e margem são **parâmetros ilustrativos** (`business_params.py`). O valor só vira argumento de investimento com dados financeiros reais; se a razão perda/margem for bem menor que 10:1, o resultado pode encolher bastante.

### Resumo por hipótese

| Hipótese | Conclusão | Status |
|---|---|---|
| H1 | O uso do limite é o principal alarme de risco | ✅ Confirmada |
| H2 | Dívida/renda importa menos que o esperado; o peso da idade precisa de checagem de viés | ❌ Contrariada |
| H3 | Dá para cortar a inadimplência em ~74% aprovando ~73% dos pedidos | ✅ Confirmada |
| H4 | O modelo se paga, com a ressalva de que os valores em reais são simulados | ✅ Confirmada |

## Perguntas de investigação
1. **Quais variáveis têm maior poder preditivo?** → respondida no ranking SHAP e em H1/H2.
2. **Existe um threshold que equilibra aprovação e inadimplência?** → respondida em H3 (0,45).
3. **O modelo mantém performance estável entre faixas de renda e idade?** → *em andamento* (notebook 02). Uma execução exploratória anterior sinalizou taxa elevada de bons pagadores negados na faixa 18–30 anos e de maus pagadores aprovados nas faixas 60+ e de renda mais alta; a análise será refeita com o threshold 0,45.
4. **Qual o impacto financeiro estimado?** → respondida em H4 (+R$ 14,5 mi, ilustrativo).

## Modelo
- **Modelo principal:** Gradient Boosting (XGBoost) — dado tabular, alvo binário desbalanceado (~6,7% de positivos) e necessidade de capturar interações não lineares entre variáveis financeiras
- **Modelo de comparação (baseline interpretável):** Regressão Logística — benchmark de interpretabilidade e piso de performance (ROC-AUC 0,788 vs. 0,867 do XGBoost)
- **Explicabilidade:** SHAP obrigatório para o modelo principal — cada decisão de aprovação/negação precisa ser rastreável até as variáveis que mais pesaram
- **Tratamento do desbalanceamento:** class weighting ou SMOTE, com decisão documentada de qual técnica foi usada e por quê

## Arquitetura no Databricks (pipeline ponta a ponta)
1. **Bronze** — ingestão via Autoloader do CSV bruto
2. **Silver** — tratamento de outliers (ex.: idade = 0, utilização de crédito > 10), imputação de nulos, padronização de schema
3. **Gold / Feature Store** — feature table versionada, pronta para treino
4. **Treino** — MLflow tracking de experimentos, comparação XGBoost vs. Regressão Logística
5. **Registro** — MLflow Model Registry no Unity Catalog (`credito_risco_xgb`), com o modelo vencedor promovido pelo alias `@champion`
6. **Produção** — inferência batch (`03_inferencia_batch.py`) pontuando os registros ainda sem rótulo
7. **Monitoramento** — notebooks de data drift e concept drift, com métricas no MLflow e histórico em tabelas Delta (detalhes abaixo)
8. **Governança** — Unity Catalog com RBAC controlando quem acessa a feature table e o endpoint

## Monitoramento pós-deploy

O monitoramento responde a duas perguntas diferentes, tratadas em notebooks separados em `notebooks/monitoracao/`. A lógica de cálculo fica em `src/monitoring/`, em numpy/pandas puro, e é coberta por testes unitários (`tests/test_data_drift.py` e `tests/test_concept_drift.py`) que rodam sem cluster.

| | Data drift | Concept drift |
|---|---|---|
| **Pergunta** | Os dados que estão chegando parecem com os dados de treino? | O modelo continua acertando, mesmo que os dados pareçam iguais? |
| **Notebook** | `01_drift_dados.py` | `02_concept_drift.py` |
| **Comparação** | Base de treino (Gold com rótulo) vs. lote de scoring (Gold sem rótulo) | AUC-ROC de referência vs. AUC-ROC do lote avaliado, usando o `@champion` |
| **Métricas** | PSI por feature (bins pelos decis da referência) + KS como segunda leitura | Queda absoluta de AUC-ROC + KS do lote atual |
| **Limiares** | PSI < 0,10 estável · 0,10–0,25 moderado · ≥ 0,25 severo | Alerta quando a AUC cai 0,03 ou mais |
| **Saída** | `ml.monitoramento_drift_dados` + experimento MLflow | `ml.monitoramento_concept_drift` + experimento MLflow |

Os limiares são pontos de partida documentados e devem ser calibrados com a área de risco.

**Por que o concept drift roda em modo simulado.** O alvo tem horizonte de 24 meses: o resultado real de um cliente pontuado hoje só é conhecido cerca de dois anos depois. Sem esses rótulos atrasados, não há como medir a performance real em produção. O notebook foi preparado para os dois cenários:

- **Modo real:** lê uma tabela de resultados realizados (`gold.resultados_realizados`) e compara com a AUC registrada como tag na versão do modelo. Essa tabela ainda não existe; quando ela não é encontrada, o notebook cai automaticamente para simulação.
- **Modo simulado:** gera um lote sintético em que o peso de `debt_ratio` e `revolving_utilization` na relação com o alvo é deliberadamente enfraquecido. Serve para validar que a lógica de alerta dispara quando deveria. O valor absoluto da AUC sintética não é comparável ao do treino; o que importa é a queda relativa.

## Limitações assumidas
- O dado é de 2011 e de clientes americanos — usado como base metodológica, não como fonte de verdade sobre o mercado brasileiro; o cruzamento com o SCR.data contextualiza essa diferença, não a corrige
- A variável-alvo tem 24 meses de horizonte; decisões de crédito com prazo diferente exigiriam reavaliação do problema
- Custo de inadimplência, margem e piso de aprovação são parâmetros ilustrativos
- O peso relevante de `age` no modelo exige análise de viés antes de qualquer uso em decisão real
- O concept drift só pode ser medido de verdade com rótulos realizados, que chegam com cerca de 24 meses de atraso; até lá, o monitoramento de performance é validado apenas em simulação e o data drift funciona como alerta antecipado

## Próximos passos
- [ ] Concluir o notebook 02 (viés por idade e renda) com o threshold 0,45 e avaliar thresholds por segmento
- [ ] Trocar Pearson por Spearman no teste de H1
- [ ] Investigar a concentração de valores repetidos em `monthly_income` na camada Silver
- [ ] Tratar features de contagem com muitos zeros no cálculo do PSI (bins por valor distinto em vez de decis)
- [ ] Criar a tabela `resultados_realizados` para ativar o modo real do concept drift
- [ ] Avaliar Model Serving para scoring em tempo real, complementando a inferência batch

## Status
Em andamento — ingestão, treino, validação das hipóteses H1–H4, registro do `@champion`, inferência batch e monitoramento de drift concluídos; análise de viés e rótulos realizados para concept drift pendentes.
