## perguntas 

Sim, no mercado de dados existe um conjunto padronizado de perguntas essenciais — frequentemente chamado de Machine Learning Project Canvas ou Discovery de Negócio — que serve para praticamente qualquer iniciativa de Machine Learning antes de escrever uma única linha de código.

1. Problema e Valor de Negócio

Qual é a decisão que o negócio precisa tomar hoje e como ela é tomada sem o modelo?

Qual métrica financeira ou de negócio queremos mover (ex.: aumento de receita, redução de inadimplência, redução de churn)?

Como será calculado o retorno sobre o investimento (ROI) deste projeto?

2. Definição do Target e Formulação Matemática

O que exatamente estamos tentando prever (uma probabilidade, uma categoria, um valor contínuo, um ranking)?

Qual é a definição exata e inequívoca do "evento" ou "sucesso/fracasso"?

Qual é a janela de observação e a janela de predição necessárias para o problema?

3. Custos de Erro e Trade-offs

O que custa mais caro para o negócio: um Falso Positivo ou um Falso Negativo?

Qual é a taxa aceitável de erros ou o limiar mínimo de tolerância para o modelo ser útil?

O modelo será usado de forma totalmente autônoma ou para apoiar um analista humano (human-in-the-loop)?

4. Dados, Histórico e Viabilidade

Os dados históricos necessários para o treino já existem e estão acessíveis?

As variáveis que pretendemos usar no treino estarão realmente disponíveis no momento da predição em tempo real (sem data leakage)?

Há histórico suficiente de dados rotulados (exemplos positivos e negativos confirmados)?

5. Operação, Latência e Arquitetura

A predição precisa ser feita em tempo real (via API, com SLA de milissegundos) ou pode ser processada em lote (batch, ex.: uma vez por dia)?

Onde o modelo será implantado e qual é o volume esperado de requisições por segundo?

Como e com que frequência os dados de entrada mudam ou se degradam?

6. Governança e Restrições Regulatórias

O modelo precisa ser 100% explicável e auditável (ex.: exigência regulatória para justificar decisões a clientes)?

Existem variáveis proibidas por lei ou conformidade (dados sensíveis, viés demográfico, etc.)?

Quem é o responsável de negócio (product owner / comitê) que valida e aprova a política final?




## diante das informações disponibilizadas quais perguntas o cientista deve fazer para o cliente

Para viabilizar a modelagem e garantir aderência aos requisitos de risco, auditoria e arquitetura, o cientista de dados deve fazer as seguintes perguntas ao cliente:   
Desconhecido

Impacto Financeiro e Políticas de Crédito

Qual é o volume financeiro médio concedido por operação e qual é a perda financeira média em R$ gerada por inadimplentes na política atual?   
Desconhecido

Qual é a taxa de aprovação atual da carteira e qual o limite máximo aceitável de perda ou inadimplência?   
Desconhecido

Qual é a regra atual para atrelar o risco à taxa de juros praticada?   
Desconhecido

Existem regras duras de corte pré-modelo que reprovam solicitações antes do cálculo do score (como faixas de idade específicas fora de 18 a 115 anos)?   
Desconhecido

Conformidade e Auditoria

O comitê de auditoria exige explicabilidade linear direta (como scorecards baseados em WoE e Regressão Logística) ou aceita modelos de gradiente (LightGBM/XGBoost) explicados por SHAP values?   
Desconhecido

É obrigatório retornar e armazenar na decisão os motivos individuais de recusa para cada cliente reprovado?   
Desconhecido

Dados e Engenharia de Produção

Como o negócio interpreta e trata operacionalmente clientes com renda ausente (monthly_income nulo) no processo manual?   
Desconhecido

Os registros com atrasos marcados com códigos atípicos (valores ≥ 96) representam erros de preenchimento, perda de contato ou clientes renegociados?   
Desconhecido

Qual é a latência máxima (SLA em milissegundos) permitida para a chamada do endpoint de decisão em produção?   
Desconhecido

O pipeline na camada Databricks Silver/Gold atualiza os dados em batch ou processa transações quase em tempo real?   
Desconhecido

Deseja aprofundar em algum desses blocos para preparar a reunião de alinhamento com a equipe de risco?



## como um cientista de dados com bases nas informações obtidas deve pensar para desenvolver o plrojeto e chegar na melhor solução?


O pensamento de um cientista de dados de alto nível não começa no algoritmo, mas no impacto financeiro e na governança do negócio. Para transformar esse problema em uma solução que alcance o topo do mercado (AUC $\ge$ 0,86) e passe em qualquer comitê de risco e auditoria, o raciocínio deve ser estruturado em seis etapas lógicas:

---

**1. Alinhamento com a Dor do Negócio**

* **Trade-off de Lucro vs. Risco**: O objetivo não é apenas prever uma probabilidade, mas decidir quem aprovar e precificar a taxa de juros.
* **O custo dos erros**: Um Falso Positivo (rejeitar um bom pagador) gera custo de oportunidade; um Falso Negativo (aprovar quem fica inadimplente) gera prejuízo direto de capital.
* **Critério de auditoria**: A decisão precisa ser explicável individualmente (por que o cliente foi recusado?) para cumprir regras regulatórias.

---

**2. Diagnóstico Crítico dos Dados (EDA & Higienização)**

* **Tratamento dos atrasos codificados (96/98)**: No Give Me Some Credit, valores como 96 ou 98 nas colunas de atraso representam códigos de sistema (ex.: erro de preenchimento ou perda de contato), e não 98 atrasos reais. A flag `has_delinquency_outlier` isola esses casos, exigindo imputação dedicada ou tratamento por binning.
* **Valores nulos com significado de negócio**: Em `monthly_income`, a ausência de renda costuma ser informativa (clientes sem renda formal comprovada). Cria-se uma variável indicadora e imputa-se a mediana condicional por faixa etária ou ocupação.
* **Outliers em `debt_ratio`**: Clientes com valores astronômicos (ex.: > 1000) geralmente têm renda mensal registrada como zero ou ausente. Tratar isso evita distorções em modelos lineares ou baseados em distância.

---

**3. Feature Engineering Estratégica (O Salto de Performance)**

Modelos de crédito ganham performance relevante com variáveis comportamentais derivadas:

* **Agregação de Atrasos**: Somatório total de ocorrências de atraso (`num_times_30_59_days_late` + `num_times_60_89_days_late` + `num_times_90_days_late`).
* **Severidade do Atraso**: Razão entre atrasos graves (90+ dias) sobre o total de atrasos.
* **Carga Financeira per Capita**: Renda mensal dividida por `(num_dependents + 1)`.
* **Exposição de Risco**: Proporção de linhas sem garantia sobre o total de linhas abertas.

---

**4. Arquitetura de Modelagem e a Escolha Padrão de Mercado**

| Abordagem | AUC Esperada | Explicabilidade | Recomendação de Mercado |
| :--- | :--- | :--- | :--- |
| **Scorecard Tradicional** (WoE + Regressão Logística) | ~0,80 – 0,82 | Total (Tabela de pontos direta) | Baseline regulatório |
| **LightGBM / XGBoost** | **0,86 – 0,87** | Alta via **SHAP Values** | **Melhor Opção** |
| **Ensemble (Stacking)** | 0,865 – 0,875 | Baixa a Média | Overkill para produção |

* **A melhor opção técnica e prática**: O **LightGBM** com calibração de probabilidade (Isotonic Regression ou Platt Scaling) é o estado da arte para crédito tabular. Ele lida nativamente com assimetrias, valores faltantes e atinge o teto de AUC da competição (~0,86–0,87), mantendo a inferência rápida em milissegundos.

---

**5. Tradução Técnica para Valor de Negócio**

* **Curva de KS (Kolmogorov-Smirnov)**: Identificar o ponto de corte onde a separação acumulada entre bons e maus pagadores é máxima.
* **Definição de Faixas de Risco (Rating de Crédito)**: Converter o score contínuo em ratings (ex.: A a F) para associar diretamente às faixas de juros.
* **Matriz de Custo Financeiro**: Calcular a economia em reais da política simulando o saldo médio concedido versus a taxa de recuperação histórica.

---

**6. Governança, Deploy e MLOps**

* **Feature Store e Pipeline**: Manter consistência lógica estrita entre o processamento Silver/Gold e o payload de entrada da API.
* **Explicabilidade em Tempo Real**: Cada resposta da API de scoring deve retornar os 3 principais fatores de recusa (Top-3 SHAP values negativos), atendendo aos requisitos de auditoria.
* **Monitoramento Contínuo**: Acompanhar o **PSI (Population Stability Index)** do score e o drift das variáveis preditoras para detectar degradação antes que a inadimplência suba na carteira real.
