# Resultados detalhados — Perguntas de investigação (Q1 a Q4)

> Os valores em R$ usam parâmetros **ilustrativos** (R$ 15.000 por mau pagador aprovado, R$ 1.500 por bom pagador negado e R$ 12.000 de valor médio por operação), definidos em `src/config/business_params.py`. Servem para demonstrar a metodologia, não como projeção de negócio.

## Modelo treinado

O modelo principal (XGBoost) obteve **AUC-ROC = 0,8641** no conjunto de teste. O AUC mede a capacidade de separar bons e maus pagadores, em uma escala de 0,5 (equivalente ao acaso) a 1,0 (perfeito). O resultado supera a meta do projeto (≥ 0,85) e fica na faixa das soluções de referência da competição original.

---

## Q1 — Quais variáveis mais pesam na decisão do modelo? (SHAP)

**Top 5 do ranking:**

| Posição | Variável | Descrição |
|---|---|---|
| 1º | `total_delinquency_events` | Total de eventos de atraso |
| 2º | `revolving_utilization` | Percentual do limite rotativo utilizado |
| 3º | `age` | Idade |
| 4º | `num_open_credit_lines` | Linhas de crédito abertas |
| 5º | `monthly_income` | Renda mensal |

### Checagem 1 — Atrasos individuais entre as 5 mais importantes: **contrariada**

O modelo prefere a variável **agregada** `total_delinquency_events` às suas partes (`num_times_30_59_days_late` e equivalentes). Estatisticamente, faz sentido: a soma concentra o mesmo sinal das variáveis separadas, então o modelo usa a soma e atribui menor importância marginal às partes. Isso **não** significa que atraso não importa; significa que a forma agregada é mais eficiente que a fatiada.

### Checagem 2 — Utilização do limite rotativo supera a renda: **confirmada**

`revolving_utilization` (2º) ficou bem à frente de `monthly_income` (5º). O quanto a pessoa já usa do limite disponível prevê inadimplência melhor do que quanto ela ganha, o que é consistente com a literatura de crédito: comportamento de uso costuma ser mais preditivo do que nível de renda isolado.

### Ponto de atenção para a hipótese H2

`age` aparece em 3º lugar, enquanto `debt_ratio` não aparece entre as 5 mais importantes. Isso é relevante para a hipótese H2 ("`debt_ratio` pesa mais que `age`") e deve ser considerado ao interpretar o veredito dela.

---

## Q2 — Qual o melhor ponto de corte para aprovar ou negar crédito?

O modelo devolve uma probabilidade (0 a 1) de inadimplência. O threshold define a partir de qual probabilidade o crédito é negado.

| Threshold | Custo total esperado |
|---|---|
| 0,50 (padrão) | R$ 18,98 milhões |
| **0,56 (ótimo)** | **R$ 18,77 milhões** |

**Economia: cerca de R$ 204 mil**, apenas pelo ajuste do corte.

Negar crédito só quando o modelo tem pelo menos 56% de confiança em inadimplência equilibra melhor os dois erros: negar um bom pagador (R$ 1.500) e aprovar um mau pagador (R$ 15.000). Como o segundo custa 10× mais, faz sentido o threshold subir um pouco. É um **ajuste fino**, não uma mudança drástica.

---

## Q3 — O modelo trata todos os grupos de forma equivalente? (viés por idade e renda)

Este é o achado mais relevante para a discussão de negócio e de compliance.

### Por idade

| Faixa | Alerta | Taxa |
|---|---|---|
| **18–30 anos** | Bons pagadores negados indevidamente (FPR) | **29,7%**, muito acima da média das faixas |
| **60+ anos** | Maus pagadores aprovados por engano (FNR) | **48,3%**, quase metade dos inadimplentes da faixa passa pelo filtro |

### Por renda

| Faixa | Alerta | Taxa |
|---|---|---|
| **Acima de R$ 8.250** | Maus pagadores aprovados por engano (FNR) | **42,5%** |

### Leitura de negócio

O modelo **não é neutro entre grupos**: é mais restritivo com jovens e mais permissivo com idosos e rendas altas.

- **Jovens:** uma hipótese plausível é o histórico de crédito mais curto (menos dados leva a um modelo mais conservador). Essa explicação **não foi testada**.
- **Renda alta:** o modelo pode associar renda alta a menor risco (razoável, mas nem sempre correto).
- **Impacto:** a inadimplência não detectada se concentra em 60+ e renda alta (desempenho da carteira), e os jovens são negados de forma desproporcional (risco regulatório e reputacional).

> **Nota de qualidade de dado:** o aviso de "apenas 4 faixas de renda" (em vez de 5) surgiu porque há concentração de valores repetidos ou baixos em `monthly_income`. Vale investigar a qualidade desse dado na Silver, pois pode distorcer a análise por renda.

---

## Q4 — Quanto o modelo economiza em relação a aprovar todos?

| Cenário | Taxa de inadimplência | Perda estimada |
|---|---|---|
| Sem modelo (aprova todos) | 6,69% | R$ 30,08 milhões |
| Com modelo (threshold ótimo) | 2,35% | R$ 8,57 milhões |

| Componente | Valor |
|---|---|
| Redução bruta de perdas | R$ 21,52 milhões |
| (−) Custo de oportunidade: 5.375 bons pagadores negados | R$ 8,06 milhões |
| **Impacto financeiro líquido estimado** | **R$ 13,45 milhões** |
| Operações analisadas | 37.500 |

Mesmo descontando o custo de negar bons pagadores, o modelo entrega cerca de **R$ 13,4 milhões de ganho líquido** na base de teste, e a inadimplência cai de 6,7% para 2,35%.

---

## Conclusão

O modelo é tecnicamente sólido e financeiramente favorável **nas premissas adotadas**. Antes de ir para produção, recomenda-se:

1. **Tratar o viés contra jovens** (Q3), que gera risco regulatório e reputacional.
2. **Investigar por que a faixa de renda alta deixa escapar tantos maus pagadores** (Q3).
3. **Avaliar threshold por segmento** em vez de um único corte global.
4. **Substituir os parâmetros financeiros ilustrativos** por valores reais do negócio.
