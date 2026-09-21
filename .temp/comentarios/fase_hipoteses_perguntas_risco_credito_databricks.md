# Fase de hipóteses e perguntas — Projeto `risco_credito_databricks`

> **Objetivo deste documento:** explicar, de forma didática, como o projeto transforma um modelo de risco de crédito em **respostas objetivas** para o negócio: "podemos confiar no modelo?", "onde cortar a aprovação?", "ele é justo?" e "quanto dinheiro ele gera?".
> **Ponto de partida:** as tabelas Gold (`dim_customer` e `fct_credit_profile`) descritas no documento da fase de ingestão.
> **Dataset:** *Give Me Some Credit* (Kaggle). Alvo: `target_dlq_2yrs` (1 = atraso de 90+ dias em 2 anos).

---

## 1. A ideia em uma imagem

Depois que os dados estão prontos na Gold, entram duas famílias de perguntas. Ambas usam o **mesmo modelo treinado**, mas respondem a coisas diferentes:

```
                      Gold (fct_credit_profile + dim_customer)
                                      │
                                      ▼
                        Modelo treinado (XGBoost) + SHAP
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
  HIPÓTESES  H1 a H4                               PERGUNTAS  Q1 a Q4
  "A minha tese se confirma?"                      "O que o modelo me diz?"
  (confirmada / contrariada)                       (respostas e números)
             │                                                 │
   H1 Rotativo pesa na inadimplência?              Q1 Quais variáveis mais pesam? (SHAP)
   H2 Dívida/renda pesa mais que idade?            Q2 Qual o threshold ótimo (custo assimétrico)?
   H3 Existe um threshold viável?                  Q3 O modelo é justo por idade e renda?
   H4 O modelo dá lucro vs. política atual?        Q4 Qual o impacto financeiro líquido (R$)?
             │                                                 │
             └────────────────────────┬────────────────────────┘
                                      ▼
                     Resultados registrados (Gold e MLflow)
                                      │
                                      ▼
                  Inferência em lote: aplica a decisão ao lote novo
```

### Qual a diferença entre hipótese e pergunta?

| | Hipótese (H) | Pergunta de investigação (Q) |
|---|---|---|
| **O que é** | Uma **tese escrita antes** de olhar o resultado | Uma dúvida aberta, respondida com dados |
| **Resultado** | Um veredito: *Confirmada* ou *Contrariada / Inconclusiva* | Uma tabela, um gráfico ou um valor |
| **Exemplo** | "Quem usa mais do limite tem mais risco de inadimplir" | "Qual o threshold que minimiza o custo?" |
| **Perigo se mal feita** | Conclusão frágil, sem critério claro | Resposta sem ligação com uma decisão |

---

## 2. Como as hipóteses e perguntas nasceram

Segundo o histórico do projeto, o caminho foi este:

1. **H1 e H2** já existiam no README e são hipóteses de **validação técnica**. Respondem "posso confiar neste modelo?".
2. Notou-se que H1 e H2 **não dizem o que fazer com o modelo** (quanto aprovar, onde cortar, quanto custa). Por isso foram acrescentadas **H3 e H4**, hipóteses de **decisão de negócio**.
3. As **Perguntas Q1 a Q4** já estavam no README como "perguntas de investigação" do Case 1: importância de variáveis (SHAP), threshold sob custo assimétrico, viés por segmento e impacto em R$.
4. Como todas dependem de **um modelo já treinado** (SHAP, probabilidades e matriz de confusão só existem depois do treino), elas ficam depois da modelagem, não na EDA. Antes do modelo só cabem "aproximações", como Information Value e distribuições por subgrupo.

### Um princípio importante

Toda hipótese deve estar ligada a **uma decisão que alguém vai tomar**:

| Hipótese/Pergunta | Decisão que ela sustenta |
|---|---|
| H1, H2 | Confio ou não confio no modelo? |
| H3, Q2 | Qual ponto de corte usar na política de crédito? |
| H4, Q4 | Vale substituir a política atual pelo modelo? |
| Q3 | Posso ir para produção sem risco jurídico ou reputacional? |

---

## 3. Como o código está organizado

O padrão é o mesmo da ingestão: **a lógica fica em `src/`; os notebooks só orquestram e plotam.**

| Notebook (`notebooks/ml/`) | Responde | Módulo com a lógica |
|---|---|---|
| `01_hipoteses_h1_h4.py` | H1, H2, H3 e H4 | `src/models/hypothesis_validation.py` |
| `02_vies_threshold_roi.py` | Q1, Q2, Q3 e Q4 | `src/models/bias_roi_validation.py` |
| `03_inferencia_batch.py` | Aplica a decisão ao lote novo | `src/ml/decisao.py` |

Há ainda um módulo compartilhado, `src/config/business_params.py`, que guarda os **valores financeiros** usados por H4 e Q2/Q4.

### Por que centralizar os valores financeiros?

Antes, H4 usava R$ 8.000 (perda por inadimplente) e R$ 1.200 (margem por bom cliente), enquanto Q2/Q4 usavam R$ 15.000 e R$ 1.500. Eram dois conjuntos de números sem motivo para divergir. Agora existe **uma única fonte de verdade**:

```
PERDA_MEDIA_POR_INADIMPLENCIA_R  = CUSTO_APROVAR_MAU_PAGADOR
MARGEM_MEDIA_POR_CLIENTE_BOM_R   = CUSTO_NEGAR_BOM_PAGADOR
```

> Os valores são **ilustrativos**. Antes de qualquer uso real, devem ser substituídos pelos números do negócio.

---

## 4. Notebook 1 — Hipóteses H1 a H4 (`01_hipoteses_h1_h4.py`)

### 4.1 Passo a passo do notebook

1. Recebe o widget `catalog` (`credito_dev`, `credito_hml` ou `credito_prd`).
2. Localiza a raiz do repositório e importa `hypothesis_validation` (`hv`).
3. **Treina** dois modelos com `hv.run_training`: **XGBoost** (o principal) e **Regressão Logística** (referência para comparação).
4. Calcula o **SHAP** (`hv.compute_shap`), que gera o ranking de importância das variáveis.
5. Valida cada hipótese com `hv.validate_h1` a `hv.validate_h4`, imprime os números e plota um gráfico por hipótese.
6. Monta um **resumo** (`build_hypotheses_summary`) e **persiste na Gold** (`persist_hypotheses_summary`).

> **O que é SHAP?** Uma técnica que mede o quanto cada variável empurrou a previsão de cada cliente para cima (mais risco) ou para baixo (menos risco). A média do valor absoluto (`mean_abs_shap`) resume a importância geral da variável.

### 4.2 H1 — Utilização de crédito rotativo

**Tese:** clientes que usam mais do limite de crédito (`revolving_utilization`) têm probabilidade maior de inadimplir em 2 anos, pois usar quase todo o limite indica pouca folga financeira.

| Item | Como é feito |
|---|---|
| **O que se mede** | Correlação entre o **valor da variável** e o **valor SHAP** dela, nos clientes da amostra |
| **Critério** | Correlação **maior que um mínimo** (na primeira versão do script, 0,05). Não basta ser positiva: precisa ter magnitude |
| **Também mostra** | A posição da variável no ranking SHAP e o seu `|SHAP|` médio |
| **Gráfico** | Dispersão (valor da variável × SHAP), com o eixo X limitado a 2,0 |
| **Veredito** | `Confirmada` ou `Contrariada / Inconclusiva` |

**Leitura simples:** se, ao subir o uso do limite, o SHAP também sobe, o modelo trata mais uso como mais risco. É exatamente o que a tese afirma.

**O que o teste NÃO prova:** *causalidade*. Mostra que o modelo usa a variável desse jeito, não que o uso do limite causa a inadimplência.

### 4.3 H2 — Dívida/renda como preditor dominante

**Tese:** o `debt_ratio` tem mais poder preditivo que a idade (`age`); ou seja, a capacidade de pagamento pesa mais que o perfil demográfico.

| Item | Como é feito |
|---|---|
| **O que se mede** | O `|SHAP|` médio de `debt_ratio` **contra** o de `age` |
| **Critério** | Diferença relativa `(shap_debt_ratio − shap_age) / shap_age` **acima de uma margem** (10% na versão de 11/09) |
| **Gráfico** | Duas barras (`debt_ratio` × `age`) com a diferença relativa no título |
| **Veredito** | `Confirmada` ou `Contrariada / Inconclusiva` |

**Por que existe uma margem?** Sem ela, uma diferença de 0,0001 já "confirmaria" a tese. A margem evita conclusões frágeis diante de um questionamento simples: "e se a diferença for ínfima?".

**O que o teste NÃO prova:** que o perfil demográfico **completo** (idade, dependentes etc.) pese menos que a dívida. Ele compara apenas duas variáveis.

### 4.4 H3 — Threshold ótimo de aprovação

**Tese:** existe um ponto de corte de probabilidade que **reduz a inadimplência da carteira aprovada** sem derrubar a taxa de aprovação abaixo de um **piso viável** para o negócio.

**Como funciona a varredura:**

```
Para cada threshold t de 0,05 a 0,90 (passo 0,05):
    aprovados  = clientes com probabilidade < t
    taxa de aprovação    = % de clientes aprovados
    inadimplência aprovados = % de inadimplentes entre os aprovados

Filtra os thresholds com taxa de aprovação >= piso (70%, ilustrativo)
Escolhe, entre eles, o de MENOR inadimplência na carteira aprovada
Compara com o baseline (aprovar todo mundo = taxa de inadimplência da base)
```

| Item | Detalhe |
|---|---|
| **Baseline** | Política atual simplificada: aprovar todos (taxa de inadimplência real da base de teste) |
| **Piso de aprovação** | 70%, valor **ilustrativo** que deve ser ajustado à política real |
| **Confirmada se** | A inadimplência na carteira aprovada ficar **abaixo do baseline** |
| **Inconclusiva se** | Nenhum threshold atingir o piso de aprovação |

**Por que o piso existe?** Porque é fácil zerar a inadimplência rejeitando todo mundo. A restrição de negócio impede essa solução inútil.

**O que o teste NÃO prova:** que o threshold é **estável no tempo**. Ele vale para a base analisada.

### 4.5 H4 — Impacto financeiro vs. política atual

**Tese:** aplicado retroativamente à base histórica, o modelo teria evitado mais perdas (em R$) do que custaria rejeitar bons pagadores.

**A conta:**

```
Perda evitada          = maus pagadores evitados   × perda média por inadimplência
Custo de oportunidade  = bons pagadores rejeitados × margem média por cliente bom
Impacto líquido        = Perda evitada − Custo de oportunidade
```

| Item | Detalhe |
|---|---|
| **Threshold usado** | **Herdado de H3** (o melhor threshold que respeita o piso de aprovação) |
| **Valores em R$** | Vêm de `src/config/business_params.py` |
| **Gráfico** | Três barras: perda evitada, custo de oportunidade e impacto líquido (verde se positivo) |
| **Sem H3 viável** | Se H3 for inconclusiva, H4 não calcula e só informa o status |

**O que o teste NÃO prova:** nada em R$ é real enquanto os valores forem ilustrativos.

### 4.6 Persistência

Ao final, o notebook monta uma tabela-resumo com as quatro hipóteses (status e métricas), exibe com `display` e grava na Gold. Assim o resultado fica **consultável e versionado**, e não apenas impresso na tela.

---

## 5. Notebook 2 — Perguntas Q1 a Q4 (`02_vies_threshold_roi.py`)

### 5.1 Passo a passo

1. Recebe o widget `catalog` e fixa `MODEL_NAME = "credito_risco_xgb"` e `MODEL_ALIAS = "champion"`.
2. Carrega os dados de treino da Gold (`load_gold_training_data`) e separa treino e teste (`prepare_train_test`).
3. Obtém o modelo com `load_or_train_model`: usa o `@champion` do Model Registry ou treina e registra um novo, conforme o nome da função indica.
4. Calcula as probabilidades do conjunto de teste (`predict_proba`) e responde Q1 a Q4 em sequência.
5. Registra tudo no **MLflow** (`log_validation_to_mlflow`).

### 5.2 Q1 — Importância de variáveis via SHAP

| Item | Detalhe |
|---|---|
| **Pergunta** | Quais variáveis mais influenciam a decisão do modelo? |
| **Saída** | Ranking em tabela, *summary plot* e gráfico de barras |
| **Checagem extra** | As variáveis de H1 e H2 aparecem entre as **5 mais importantes**? (`check_top_n`) |

### 5.3 Q2 — Threshold ótimo sob custo assimétrico

**A ideia central:** errar não custa igual nos dois sentidos.

| Erro | O que acontece | Custo |
|---|---|---|
| Aprovar um **mau** pagador | O cliente não paga | **Alto** (`CUSTO_APROVAR_MAU_PAGADOR`) |
| Negar um **bom** pagador | Perde-se a margem da venda | **Menor** (`CUSTO_NEGAR_BOM_PAGADOR`) |

Como aprovar um mau pagador custa muito mais, o threshold que minimiza o custo **não é 0,50**. O notebook varre os thresholds, calcula o custo total esperado em cada um e mostra:

- o threshold ótimo;
- o custo nele e o custo no threshold 0,50;
- a **economia** entre os dois;
- o gráfico "custo esperado × threshold", com o ótimo destacado.

### 5.4 Q3 — Estabilidade e viés por idade e renda

| Item | Detalhe |
|---|---|
| **Pergunta** | O modelo erra de forma desproporcional para algum grupo? |
| **Como** | Divide os clientes em **faixas de idade** e **faixas de renda** e calcula o desempenho de cada grupo, usando o threshold ótimo de Q2 |
| **Alerta** | Sinaliza grupos com **FPR/FNR acima da média + 1 desvio-padrão** |
| **Aviso de renda** | Emite `aviso_renda` quando há algo a observar na variável de renda (por exemplo, muitos nulos) |

> **FPR** (falso positivo): bom pagador negado por engano. **FNR** (falso negativo): mau pagador aprovado por engano.

**Por que importa:** um modelo pode ter ótima métrica geral e ainda assim penalizar um grupo. Em crédito, isso é risco de **discriminação indireta**.

### 5.5 Q4 — Impacto financeiro estimado

Compara **sem modelo** (aprovar todos) com **com modelo** (usando o threshold de Q2):

| Indicador | Significado |
|---|---|
| Taxa de inadimplência sem/com modelo | Antes e depois |
| Perda estimada sem/com modelo | Em R$ |
| Redução bruta de inadimplência | Quanto de perda o modelo evita |
| Custo de oportunidade | Bons pagadores negados |
| **Impacto líquido** | Redução bruta − custo de oportunidade |

### 5.6 H4 e Q4 não precisam dar o mesmo número

Isso é **intencional**. Elas respondem a perguntas diferentes:

| | H4 | Q4 |
|---|---|---|
| **Threshold usado** | O de **H3** (menor inadimplência respeitando o piso de aprovação) | O de **Q2** (menor custo total) |
| **Pergunta** | "Vale trocar a política, mantendo o volume de aprovação?" | "Qual o ganho se eu cortar no ponto de menor custo?" |

---

## 6. Da resposta à decisão — Notebook 3 (`03_inferencia_batch.py`)

Os resultados acima só têm valor se a decisão chegar aos clientes. Este notebook aplica o modelo ao **lote novo** (o dataset de scoring).

```
1. Carrega o modelo credito_risco_xgb@champion do Unity Catalog
2. Busca na Gold os clientes com target_dlq_2yrs NULO (o lote de scoring)
3. Calcula a probabilidade de inadimplência de cada um
4. Aplica a regra de decisão com o threshold de aprovação
5. Grava em gold.credito_score_predictions (append, com data/hora da execução)
```

### A regra de decisão (`src/ml/decisao.py`)

```python
def classificar_decisao(probabilidade, threshold):
    # 'negado' se probabilidade >= threshold; 'aprovado' caso contrário
```

- É **Python puro** (sem Spark nem MLflow), para ser testada com `pytest` sem cluster.
- Lança `ValueError` se a probabilidade estiver fora de [0, 1].
- É coerente com H3: lá, **aprovado = probabilidade abaixo do corte**.

### O threshold de produção

O notebook recebe o widget `threshold_aprovacao` (padrão **0,56**), referenciado no código como "Q2 do README". Cada linha gravada leva o modelo, a versão e o threshold usados, o que permite **auditar** depois qual regra gerou cada decisão.

### Detalhes de engenharia que valem ser lembrados

- O modelo é carregado com `mlflow.xgboost.load_model` e **não** com `pyfunc`, porque dependendo da versão o `pyfunc` pode devolver a classe (0/1) em vez da probabilidade.
- Se não houver registros a pontuar, o notebook **encerra com aviso**, sem erro.
- Se faltar o `@champion`, ele falha com mensagem orientando a rodar antes o `02_vies_threshold_roi.py`.
- Nulos nas variáveis do lote geram **aviso** (o XGBoost lida com nulos, mas convém checar a origem).

---

## 7. Ordem de execução

```
Gold pronta (ingestão concluída)
   │
   ├─ 01_hipoteses_h1_h4.py       (treina, valida H1-H4 e grava o resumo)
   │
   ├─ 02_vies_threshold_roi.py    (responde Q1-Q4 e registra o @champion)
   │
   └─ 03_inferencia_batch.py      (precisa do @champion; aplica ao lote novo)
```

O `03` **depende** do `02`, pois é ele quem deixa o modelo `@champion` disponível.

---

## 8. Pontos de atenção (para revisar)

1. **H1 e H2 têm definições diferentes entre os dois notebooks.** No `01`, H1 é "utilização de crédito rotativo" e H2 é "`debt_ratio` vs `age`". No `02`, a checagem rápida trata H1 como "atrasos específicos no top-5" e H2 como "`revolving_utilization` / `monthly_income` no top-5". Vale padronizar para que **H1 e H2 signifiquem uma só coisa** em todo o projeto.
2. **Método de H2 mudou ao longo das conversas.** Em uma sessão, H2 foi analisada comparando `debt_ratio` contra o **perfil demográfico completo** (`age` + `num_dependents`) por *permutation importance*, com conclusão "Refutada". No código atual, a comparação é apenas `debt_ratio` × `age` por `|SHAP|`. As duas versões podem levar a vereditos diferentes.
3. **Margens e pisos são escolhas, não fatos.** O mínimo de correlação de H1, a margem de 10% de H2 e o piso de 70% de H3 foram definidos por convenção. Na mesma discussão, você observou que não existe um limiar formal para H2. O ideal é registrar de onde vem cada número ou justificar como decisão de negócio.
4. **Valores financeiros ilustrativos.** Todo resultado em R$ (H4, Q2 e Q4) depende de números que ainda não vieram do negócio.
5. **Threshold de produção definido à mão.** O `0.56` do `03` é um valor digitado no widget. Poderia vir de `business_params.py` (ou do resultado de Q2 registrado no MLflow), para não divergir do que foi calculado.
6. **H3/H4 usam o conjunto de teste para escolher e para avaliar.** O threshold é escolhido nos mesmos dados em que o resultado é medido, o que tende a deixar o ganho ligeiramente otimista. Para uma versão mais robusta, escolha o threshold em validação e reporte em teste.

---

## 9. Glossário rápido

| Termo | Significado simples |
|---|---|
| **Hipótese** | Tese escrita antes do teste, com critério claro de confirmação |
| **SHAP** | Mede quanto cada variável empurrou a previsão de cada cliente |
| **`mean_abs_shap`** | Importância geral da variável (média do valor absoluto do SHAP) |
| **Threshold** | Ponto de corte: acima dele o cliente é negado, abaixo é aprovado |
| **Taxa de aprovação** | % de clientes aprovados |
| **Baseline** | Cenário de comparação (aqui, aprovar todo mundo) |
| **Custo assimétrico** | Situação em que os dois tipos de erro têm custos diferentes |
| **FPR / FNR** | Taxa de falsos positivos (bom negado) / falsos negativos (mau aprovado) |
| **Custo de oportunidade** | Margem perdida ao negar bons pagadores |
| **`@champion`** | Alias que aponta a versão do modelo aprovada para uso no Unity Catalog |
| **Inferência em lote** | Pontuar de uma vez um conjunto de clientes novos |
| **Model Registry** | Catálogo versionado de modelos (aqui, no Unity Catalog) |
