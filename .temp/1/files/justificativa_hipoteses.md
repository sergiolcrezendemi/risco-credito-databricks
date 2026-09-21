# Por que essas 4 hipóteses — e o que elas entregam para a decisão

## O raciocínio por trás da escolha

Antes de treinar qualquer modelo, foram definidas 4 hipóteses que o projeto precisava responder. Não por curiosidade estatística, mas porque cada uma sustenta uma decisão diferente que a instituição precisará tomar caso decida usar o modelo na prática.

As quatro seguem uma progressão lógica, em duas camadas:

**Camada 1 — Confiança no modelo (H1 e H2):** o modelo está aprendendo algo que faz sentido de negócio, ou é uma caixa-preta que "acerta" por motivos que ninguém consegue explicar?

**Camada 2 — Decisão operacional (H3 e H4):** dado que o modelo é confiável, o que fazer com ele na prática? Que corte usar, e vale a pena financeiramente?

Essa ordem não é acidental: não faz sentido decidir threshold ou calcular impacto em R$ (Camada 2) de um modelo em que ainda não se confia (Camada 1). Cada hipótese resolve um risco específico antes de avançar para a próxima.

---

## H1 — O modelo reage a endividamento real, não a ruído

**A pergunta:** clientes que já usam a maior parte do limite de crédito têm mais risco de inadimplência?

**Por que essa pergunta importa:** essa é a variável mais intuitiva de risco de crédito que existe. Se o modelo não capturar esse padrão, é sinal de que algo está errado no dado ou no treinamento, antes mesmo de olhar qualquer resultado mais sofisticado.

**O que o negócio ganha com a resposta:**
- **Se confirmada:** primeira evidência concreta de que o modelo "pensa" como um analista de crédito pensaria, reduzindo o risco de aprovar um modelo que decide por motivos arbitrários.
- **Se não confirmada:** alerta cedo de que há um problema no dado ou na engenharia de features, evitando colocar em produção um modelo com defeito silencioso.

---

## H2 — Comportamento financeiro pesa mais do que perfil demográfico

**A pergunta:** a situação financeira atual do cliente (quanto ele deve em relação ao que ganha) prediz melhor do que a idade dele sozinha?

**Por que essa pergunta importa:** é uma proteção contra um risco sério. Se o modelo decidisse principalmente com base em idade (uma característica demográfica), a instituição estaria exposta a um risco reputacional e potencialmente jurídico de discriminar por perfil, em vez de avaliar capacidade de pagamento real.

**O que o negócio ganha com a resposta:**
- **Se confirmada:** uma defesa objetiva e documentada de que o modelo decide com base em comportamento financeiro, não em perfil do cliente. É um argumento importante em auditoria interna, compliance ou questionamento externo sobre critérios de aprovação.
- Reforça a Camada 1: mais um sinal de que o modelo é confiável antes de operacionalizá-lo.

---

## H3 — Existe um ponto de corte que funciona na prática

**A pergunta:** dá para usar esse modelo para aprovar ou negar crédito de um jeito que reduza inadimplência sem travar o negócio (aprovando poucos clientes demais)?

**Por que essa pergunta importa:** um modelo tecnicamente bom, mas sem um ponto de corte definido, não é operacional. Ninguém aprova crédito com "uma probabilidade": alguém aprova ou nega. Essa hipótese converte o modelo em uma regra de decisão concreta.

**O que o negócio ganha com a resposta:**
- Um número (o threshold) que a área de crédito pode efetivamente usar no dia a dia.
- Visibilidade do trade-off real: quanto se ganha em redução de inadimplência versus quanto se abre mão em volume de aprovação. Essa decisão pode hoje estar sendo tomada sem embasamento quantitativo.

---

## H4 — O modelo compensa financeiramente

**A pergunta:** o dinheiro economizado ao evitar maus pagadores é maior do que o dinheiro perdido ao, por engano, recusar bons pagadores?

**Por que essa pergunta importa:** é a pergunta que qualquer decisão de investimento em tecnologia precisa responder. Reduzir inadimplência não vale a pena se, no processo, a instituição perder ainda mais receita rejeitando clientes bons. Essa hipótese traduz todo o trabalho técnico anterior em R$, a linguagem que sustenta uma decisão de negócio.

**O que o negócio ganha com a resposta:**
- Uma estimativa de impacto financeiro líquido: o número que justifica (ou não) o investimento em colocar o modelo em produção.
- Uma base quantitativa para comparar "manter a política atual" com "adotar o modelo", em vez de uma decisão baseada em intuição.

---

## Resumo para a decisão final

| Hipótese | Pergunta que resolve | Decisão que sustenta |
|---|---|---|
| H1 | O modelo reage a endividamento real? | Confiar ou não no modelo |
| H2 | O modelo evita viés demográfico? | Aprovar o modelo em compliance/auditoria |
| H3 | Qual o ponto de corte prático? | Definir a regra operacional de aprovação |
| H4 | Vale a pena financeiramente? | Aprovar (ou não) o investimento em produção |

Juntas, as quatro hipóteses cobrem o caminho completo entre "o modelo funciona tecnicamente" e "a instituição deve usar esse modelo, e de que forma", que é exatamente a pergunta que importa para quem decide, não para quem constrói o modelo.
