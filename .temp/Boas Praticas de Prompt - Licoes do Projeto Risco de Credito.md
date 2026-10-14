prompt = claude.ia

# Boas Práticas de Prompt — Lições do Projeto Risco de Crédito

Sep 25, 2026 · @185233

Feedback sobre as idas e vindas do projeto risco\_credito\_databricks, traduzido em práticas de prompt para ser mais assertivo nos próximos projetos.

## O padrão de idas e vindas observado

Olhando o histórico completo do projeto, as idas e vindas não vieram de falta de conhecimento técnico — vieram de decisões que foram tomadas tarde demais, ou reabertas depois de já terem sido tomadas. Quatro padrões se repetem:

1. **Framework definido depois de já ter produto parcial.** H1 e H2 nasceram prontas (do README), mas só depois, numa conversa separada, percebeu-se que faltavam hipóteses que respondessem "o que eu faço?" — e H3/H4 foram propostas a posteriori.
2. **A mesma pergunta de método foi feita mais de uma vez, em conversas diferentes.** "Como devo pensar diante da base para criar hipóteses?", "existe uma regra de ouro?", "devo responder hipótese ou pergunta primeiro?" — todas variações da mesma questão de fundo, revisitada em sessões separadas em vez de resolvida uma vez e reaproveitada.
3. **Parâmetros e textos mudaram sem uma fonte única de verdade.** Os valores financeiros de H4 (custo de aprovar mau pagador, margem de bom pagador) chegaram a existir em duas versões diferentes em dois módulos; o enunciado de H1 e H2 mudou de redação entre conversas.
4. **Bugs de pipeline descobertos por acidente, não por especificação prévia.** A colisão de IDs entre treino e scoring, a instabilidade do teste de Pearson e os dois modelos usados como se fossem um só só apareceram porque alguém notou um número estranho — não porque o prompt original pedia esse tipo de verificação.

O fio comum: quase todo retrabalho veio de uma decisão de **método** (não de modelagem) que deveria ter sido fechada antes da primeira linha de código, e não foi.

## Incidente → causa no prompt → prática recomendada

| Incidente | Causa raiz no prompt | Prática para o próximo projeto |
| --- | --- | --- |
| H3/H4 propostas só depois de perceber a lacuna | O prompt inicial não pediu que as hipóteses fossem classificadas por papel antes de serem geradas | Na primeira mensagem, exigir que toda hipótese seja marcada como "confiança no modelo" ou "decisão operacional" — e pedir as duas categorias completas antes de testar qualquer uma |
| Parâmetros financeiros divergentes em dois módulos | Não foi exigido um arquivo único de configuração desde o início | Instruir explicitamente: "todo parâmetro usado em mais de um lugar do código mora em um único arquivo, nunca duplicado" |
| Enunciado de H1/H2 mudou de versão em versão | Hipótese não foi "congelada" com critério de erro antes de rodar o teste | Pedir que cada hipótese seja escrita, com critério de erro, antes do primeiro teste — e que qualquer mudança depois disso seja registrada como mudança, não reescrita silenciosa |
| Ambiguidade sobre o que contava como "variável de atraso" em H1 | O enunciado ficou aberto demais, sem apontar a coluna exata | Exigir que toda hipótese diga, em termos de coluna do dicionário de dados, exatamente o que está sendo testado |
| Colisão silenciosa de IDs entre treino e scoring | Não foi pedido teste de unicidade de chave na primeira versão do pipeline | Em qualquer prompt que una tabelas, incluir a exigência de um teste de unicidade da chave, desde a primeira versão |
| Correlação de Pearson distorcida por outliers, oscilando entre execuções | Não foi pedida justificativa da métrica estatística escolhida, considerando a distribuição da variável | Para qualquer teste estatístico, pedir que a escolha do método (Pearson vs. Spearman, por exemplo) seja justificada pela distribuição dos dados, não assumida por padrão |
| Dois modelos usados como se fossem um só | Faltou, desde o início, uma trava de reprodutibilidade entre o modelo registrado e o modelo avaliado | Exigir, desde o primeiro prompt de MLOps, que toda validação recalcule e compare a métrica do modelo com a registrada, travando a execução se divergir |
| A mesma pergunta de método feita em conversas separadas | Faltou um prompt-mestre reunindo o raciocínio de projeto antes de começar a codificar | Consolidar essas perguntas de método numa única conversa de planejamento, transformar a resposta num documento de referência, e citar esse documento nas conversas seguintes do projeto |

## Como estruturar o prompt inicial, antes de qualquer código

No projeto de risco de crédito, o documento de alinhamento com o "CEO" — decisão em uma frase, escopo, critério de sucesso, premissas financeiras, riscos conhecidos — só foi montado depois de já ter modelo rodando. Ele deveria ter sido o próprio primeiro prompt do projeto, não um documento posterior. Seis elementos que o primeiro prompt de um projeto de dados deveria trazer:

1. **A decisão em uma frase.** Que decisão esse projeto sustenta, e quem a toma hoje sem o modelo.
2. **Escopo explícito.** O que entra no projeto e o que fica para depois — e, para o que fica de fora, dizer isso de forma explícita em vez de deixar implícito.
3. **Critério de sucesso com números, ou marcado como pendente.** Meta técnica (ex.: AUC ≥ 0,85) e meta de negócio (redução de inadimplência, ganho em R$) — quando o número real ainda não existe, pedir que ele seja marcado como "ilustrativo, a validar", nunca como definitivo.
4. **Fonte única de verdade para parâmetros.** Pedir, desde a primeira mensagem, que todo parâmetro reutilizado (custos, thresholds, limiares) viva num único lugar do código.
5. **Padrão de teste esperado.** Que tipo de garantia automática o projeto deve ter desde o início — testes de unicidade de chave, testes de reprodutibilidade de modelo, testes de estabilidade de métrica — em vez de descobrir a necessidade deles via bug.
6. **Riscos conhecidos, nomeados antes de começar.** Se o projeto usa uma base pública como proxy, se há viés de seleção, se falta uma variável temporal — nomear essas limitações no prompt inicial evita que elas apareçam como descoberta tardia.

## Como escrever hipóteses desde a primeira mensagem

O projeto chegou a essa regra, mas só depois de já ter testado hipóteses mal escritas: uma hipótese só é boa se a resposta, seja qual for, muda o que alguém vai fazer amanhã. Se não muda, é curiosidade estatística, não hipótese de negócio. Quatro passos, na ordem certa, pedidos já no primeiro prompt:

1. **Domain prior antes do dado.** Peça para listar o que um especialista do domínio já esperaria ver, antes de abrir a base — isso evita hipótese que nasce de correlação espúria.
2. **Hipótese com quatro partes, escrita de uma vez.** Palpite, teste, critério de erro definido antes do resultado, e a ação em cada cenário de resposta. As quatro juntas, na mesma mensagem — nunca o palpite isolado, para completar depois.
3. **Ligar ao dicionário de dados só depois.** Só depois da hipótese escrita, apontar exatamente qual coluna materializa cada parte dela — isso evita ambiguidade tipo "o que conta como variável de atraso".
4. **Congelar antes de testar.** Pedir a data e o texto final antes do primeiro teste. Qualquer mudança depois disso é uma decisão nova, registrada como tal — nunca uma reescrita silenciosa do enunciado original.

## Prompt-modelo reutilizável

Um rascunho para colar como primeira mensagem do próximo projeto, ajustando o que está entre colchetes:

```
Quero construir [nome do projeto], que sustenta a seguinte decisão de
negócio: [decisão em uma frase — o que alguém decide, com base nisso].

Escopo: entra [o que entra]. Fica de fora, por enquanto: [o que fica
para depois]. Métrica técnica alvo: [ex. AUC ≥ 0,85]. Métrica de
negócio: [ex. redução de X% na inadimplência] — se o número real ainda
não existe, marque como "ilustrativo, a validar", nunca como
definitivo.

Antes de qualquer código, quero fechar:
1. As hipóteses, cada uma com palpite + teste + critério de erro +
   decisão que muda, escritas e congeladas antes do primeiro teste.
2. Um único arquivo de configuração para todo parâmetro reutilizado
   (custos, thresholds, limiares) — nunca duplicado em módulos
   diferentes.
3. Testes automáticos desde a primeira versão: unicidade de chave em
   qualquer união de tabelas, reprodutibilidade do modelo registrado
   (recalcular e comparar a métrica antes de confiar nela), e
   validação de qualquer teste estatístico contra a distribuição real
   da variável.
4. Riscos e limitações conhecidos, nomeados agora — [ex. base pública
   como proxy, viés de seleção, ausência de variável temporal].

Só depois disso, quero começar a arquitetura e o código.
```

## Checklist rápido antes de começar a codificar

- [ ] A decisão de negócio está escrita em uma frase?
- [ ] O escopo diz claramente o que entra e o que fica para depois?
- [ ] Toda meta técnica e de negócio tem número — ou está marcada como "ilustrativa, a validar"?
- [ ] Cada hipótese tem palpite, teste, critério de erro e decisão associada, escritos juntos e de uma vez?
- [ ] Existe um único lugar no código para parâmetros reutilizados?
- [ ] Testes de unicidade de chave e de reprodutibilidade de modelo estão pedidos desde a primeira versão, não como correção depois de um bug?
- [ ] Os riscos e limitações conhecidos foram nomeados antes de começar, não descobertos no meio do caminho?
- [ ] Perguntas de método ("como devo pensar diante disso") foram resolvidas uma única vez, num só lugar — não reabertas em conversas separadas?
