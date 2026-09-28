# Roteiro de Entrevista — Projeto Risco de Crédito (Databricks)

Sep 25, 2026 · @185233

Guia para explicar o projeto de risco de crédito ponta a ponta em entrevista: o raciocínio por trás de cada decisão, os problemas encontrados, as alternativas consideradas e os resultados.

## Contexto e problema de negócio

Uma instituição financeira decide, para cada pedido de crédito, se aprova a operação e com qual taxa de juros. A decisão precisa ser rápida, igual entre analistas e defensável em auditoria.

Antes de tocar no dado, foi montado um documento de alinhamento — no formato de um pitch para o CEO — para travar o que "dar certo" significa: qual decisão o projeto sustenta, o que entra e o que fica para depois (a taxa de juros exige probabilidade calibrada e ficou marcada como decisão em aberto), e quais premissas financeiras precisam ser validadas com Risco e Finanças antes de virarem número final.

Base de dados: Give Me Some Credit (Kaggle) — 150.000 clientes de treino, 10 variáveis explicativas e o alvo SeriousDlqin2yrs (inadimplência em 90+ dias nos próximos 2 anos); 101.503 clientes de scoring, sem rótulo, usados como lote de produção.

Meta técnica: AUC-ROC ≥ 0,85 — atingida em 0,866. Mas a régua de "melhor solução" não parou aí: calibração das probabilidades, ausência de viés relevante entre grupos, impacto positivo em R$ mesmo em cenário pessimista, explicabilidade por decisão (SHAP) e reprodutibilidade também entraram como critério, porque um modelo só bom em AUC não passa num comitê de risco.

## Como o pipeline foi pensado

Arquitetura em três camadas no Databricks, com Unity Catalog controlando acesso a tabelas e modelo:

- Bronze — ingestão incremental via Auto Loader, tabelas separadas para treino e scoring. Pergunta orientadora: não "como eu leio o CSV", mas "como esse dado chega de forma contínua quando isso estiver em produção".
- Silver — limpeza e padronização com quarentena lógica: idade fora de 18–120 anos e códigos de erro nos atrasos (valores como 96/98, que são flags de sistema, não contagens reais) são marcados, não descartados.
- Gold — modelo estrela (dim\_customer + fct\_credit\_profile), treino e scoring na mesma tabela (alvo nulo = lote novo).

Duas decisões foram registradas antes de ver qualquer resultado, para não virarem escolhas pós-fato:

1. Baseline interpretável primeiro. Regressão logística treinada e comparada ao XGBoost com o critério já definido: se o ganho de AUC do boosting fosse pequeno, a logística venceria por ser mais barata de auditar.
2. Split único e verificado. 25% de teste, seed 42, estratificado — usado pelos dois notebooks de validação, com checagem de que a AUC recalculada bate com a registrada no treino.

O restante do pipeline segue o mesmo fio condutor: MLflow para reprodutibilidade (hiperparâmetros, métricas e versão dos dados registrados por execução), registro no Model Registry com alias @champion apontando para a versão em produção, inferência batch consumindo esse alias, e monitoramento fechando o ciclo — porque o mundo muda depois que o modelo vai para produção.

## As quatro hipóteses

As hipóteses seguem uma regra simples, definida antes de olhar os dados: uma hipótese só é boa se a resposta, seja qual for, muda o que alguém vai fazer amanhã — senão é curiosidade estatística, não decisão de negócio. Por isso elas foram organizadas em duas camadas.

**Camada 1 — Confiança no modelo:** o modelo aprende algo que faz sentido de negócio?

| Hipótese | Enunciado | Resultado | Status |
| --- | --- | --- | --- |
| H1 | Uso do rotativo perto do limite aumenta a inadimplência | Spearman de 0,879 entre a feature e seu próprio SHAP | Confirmada |
| H2 | Dívida/renda pesa mais que idade | \|SHAP\| da idade (0,245) fica 1,6x acima de dívida/renda (0,153) | Contrariada |

**Camada 2 — Decisão operacional:** dado que o modelo é confiável, que corte usar e se vale a pena.

| Hipótese | Enunciado | Resultado | Status |
| --- | --- | --- | --- |
| H3 | Existe um threshold que reduz a inadimplência sem derrubar a aprovação | Corte em 0,45: inadimplência 6,68% → 1,76%, aprovação 73,1% | Confirmada |
| H4 | O modelo evita mais perdas em R$ do que custa recusar bons pagadores | Ponto de equilíbrio: 4,0x a margem de um bom cliente | Confirmada |

H2 foi a mais reveladora: a hipótese original apostava em dívida/renda, mas a idade pesou quase o dobro. Esse resultado inesperado foi o gatilho para a análise de viés por faixa etária, na próxima seção.

Vale contar em entrevista como as hipóteses nasceram: H1 e H2 vieram de um domain prior clássico de crédito — um analista experiente já desconfia de uso de limite e de endividamento antes de abrir a base. H3 e H4 foram propostas depois, ao notar que H1/H2 respondiam "posso confiar no modelo?" mas nenhuma respondia "o que eu faço com ele?".

Duas fragilidades foram encontradas e corrigidas no processo — vale citar como autocrítica, não como falha escondida: os parâmetros financeiros de H4 chegaram a ter dois valores diferentes em dois módulos do código, sem justificativa de negócio; e o enunciado de H1 tinha ambiguidade sobre o que contava como "variável de atraso". Ambas foram corrigidas centralizando a fonte de verdade e reescrevendo o critério antes de rodar de novo.

## Viés, calibração e comparação com a baseline

A análise de viés confirmou o que H2 sinalizou: um bom pagador de 18 a 30 anos é recusado 5,4 vezes mais que um de 60+ anos (44,5% contra 8,2% de bons pagadores negados).

Parte é risco real — a inadimplência observada nos jovens é de fato maior (12,0% contra 2,8% nos 60+). Mas quando duas populações têm taxas reais de inadimplência muito diferentes, nenhum score consegue ser bem calibrado e, ao mesmo tempo, ter a mesma taxa de bons pagadores negados nos dois grupos — é um resultado matemático (Kleinberg et al., 2016; Chouldechova, 2017), não uma falha de engenharia. A decisão sobre qual critério de justiça priorizar precisa ser tomada com risco, jurídico e compliance, não só com o time técnico.

Calibração: o modelo foi treinado com scale\_pos\_weight para compensar o desbalanceamento (6,7% de inadimplentes), o que infla as probabilidades — o score bruto previa 31,5% de inadimplência média para uma base com 6,7% observada. Isso não afeta aprovação ou negação, porque só a ordenação importa. Mas para precificar juros por risco, é obrigatório usar a versão calibrada: a calibração isotônica derrubou o erro médio de calibração de 0,248 para 0,007.

Comparação com a regressão logística, na mesma taxa de aprovação (73,1%):

| Modelo | AUC | Inadimplência da carteira | Maus recusados | Bons recusados |
| --- | --- | --- | --- | --- |
| XGBoost (@champion) | 0,866 | 1,76% | 2.023 | 8.052 |
| Regressão logística | 0,790 | 3,08% | 1.662 | 8.413 |

O XGBoost recusa 361 maus pagadores a mais e 361 bons pagadores a menos que a logística — essa é a comparação que interessa ao negócio, contra um modelo mais simples e mais fácil de explicar, não contra "aprovar todo mundo".

## Erros encontrados e como foram resolvidos

Esta é a parte que mais separa um notebook de estudo de um pipeline de produção — nenhum desses problemas aparece olhando só a métrica final (AUC).

**1. Correlação instável em H1.** A primeira versão do teste usava Pearson entre revolving\_utilization e seu SHAP. A variável é um percentual (0 a 1), mas alguns registros tinham erro de carga na casa dos milhares — outliers que distorciam Pearson. O resultado oscilava entre execuções (0,071 e 0,047), com H1 passando e falhando sem nenhuma mudança real no dado. Trocado por Spearman, que mede a relação pela ordem e não pela distância — resultado estável em 0,879 — e um teste unitário passou a reproduzir o problema para ele nunca mais voltar.

**2. Colisão silenciosa de IDs.** Os CSVs de treino e de scoring numeram clientes a partir de 1. Sem tratamento, um cliente de treino e um de scoring podiam ter o mesmo ID e virar um único registro na dimensão Gold — trocando idade e outros atributos entre as duas populações, sem gerar nenhum erro. Afetava os 101.503 clientes do lote de scoring. Corrigido com deslocamento de IDs antes da união e um teste de unicidade garantindo 251.503 clientes sem colisão.

**3. Dois modelos apresentados como um só.** Em algum momento, números do projeto vieram de execuções com versões diferentes do modelo, sem que a AUC sozinha denunciasse a diferença — os números pareciam coerentes, mas não eram comparáveis entre si. Corrigido travando os notebooks de validação: a AUC recalculada no teste precisa bater exatamente com a registrada no run de treino do @champion, senão a execução para.

**4. Parâmetros financeiros divergentes.** O custo de aprovar um mau pagador e a margem perdida ao negar um bom pagador estavam declarados em dois módulos do código, com valores diferentes (R$ 8.000/R$ 1.200 num, R$ 15.000/R$ 1.500 no outro) — não por decisão de negócio, só porque foram escritos em dois lugares. Centralizados num único arquivo (business\_params.py), hoje fonte única de verdade para H4 e para a comparação de threshold.

**5. Falha de infraestrutura na ingestão.** O pipeline Bronze/Silver quebrava com erro de volume não encontrado porque o checkpoint do streaming era referenciado antes do volume existir. Corrigido garantindo schema e volumes antes de iniciar o writeStream.

**6. Ambiguidade no carregamento do modelo.** Carregar o @champion pelo flavor genérico do MLflow (pyfunc) podia devolver a classe prevista (0/1) em vez da probabilidade, dependendo da versão instalada. Trocado pelo flavor nativo do XGBoost, que sempre devolve probabilidade sem ambiguidade — decisão pequena, mas evita um erro silencioso de calibração em produção.

## Monitoramento em produção e resultado final

O monitoramento responde duas perguntas diferentes, em notebooks separados:

|  | Data drift | Concept drift |
| --- | --- | --- |
| Pergunta | Os dados que chegam parecem com o treino? | O modelo continua acertando? |
| Métrica | PSI + KS por feature | Queda de AUC-ROC |
| Resultado no projeto | 11 de 11 features estáveis (101.503 clientes) | Sem alarme em intensidade 0; dispara a partir de 25% de drift simulado |

O concept drift roda em modo simulado porque o alvo (SeriousDlqin2yrs) só se confirma 24 meses depois — sem rótulos realizados ainda, não há como medir performance real. O notebook já está pronto para trocar para modo real assim que a tabela de resultados realizados existir; até lá, o data drift funciona como alerta antecipado.

Duas limitações foram documentadas em vej de escondidas: o PSI colapsa (retorna 0 por construção) em duas features de atraso quase todas zero, então um drift real nelas passaria despercebido — mitigado, por ora, pelo KS como segunda leitura; e a primeira versão do gerador de drift simulado não disparava alerta porque mexia numa variável fraca do modelo (debt\_ratio) — corrigido atuando nos sinais que o modelo de fato usa (uso do rotativo e atrasos).

Resultado final de negócio: o corte em 0,45 reduz a inadimplência da carteira de 6,68% para 1,76% (queda de \~74%) mantendo 73,1% de aprovação, com um ponto de equilíbrio financeiro de 4,0x a margem de um bom cliente — conclusão que não depende dos valores ilustrativos usados para simular o impacto em reais.

## Roteiro de fala para a entrevista

**Pitch de 30 segundos:** "Construí, ponta a ponta no Databricks, um score de inadimplência para decisão de crédito — da ingestão até o monitoramento em produção. O modelo chegou a 0,866 de AUC, mas o trabalho real foi mostrar que dá para usar isso com responsabilidade: há um viés etário mensurável na aprovação, a probabilidade precisa ser calibrada antes de precificar, e o monitoramento cobre os dois lados, dado e performance, depois do deploy."

**Pitch de 2 minutos**, na ordem que costuma prender a atenção de um entrevistador técnico:

1. O problema de negócio e por que a decisão precisa ser rápida, consistente e defensável em auditoria.
2. A arquitetura Bronze/Silver/Gold e a escolha de comparar XGBoost com uma baseline interpretável antes de assumir a complexidade extra.
3. As quatro hipóteses, com destaque para H2 — o achado inesperado que levou à análise de viés.
4. O viés por idade e por que ele não se resolve só com técnica.
5. Um ou dois erros do processo — a colisão de IDs ou os dois modelos conflitados são os mais fortes, porque mostram que você testa e desconfia do próprio pipeline.
6. O resultado: inadimplência caindo de 6,68% para 1,76%, monitoramento validado nos dois sentidos.

**Perguntas prováveis e como responder:**

| Pergunta do entrevistador | Como responder |
| --- | --- |
| Por que XGBoost e não só regressão logística? | O critério foi definido antes do resultado: só valeria a complexidade extra se o ganho fosse grande. Foi: AUC 0,866 vs 0,790, e inadimplência 1,76% vs 3,08% na mesma aprovação. |
| Como vocês tratam o viés encontrado? | Não é escondido nem "resolvido" tecnicamente — parte é risco real, parte é viés residual, e zerar essa diferença exige abrir mão de outro critério de justiça. Isso vira decisão de negócio, jurídico e compliance. |
| Qual foi o maior erro que vocês cometeram? | A colisão silenciosa de IDs entre treino e scoring, que trocava idade de cerca de 100 mil clientes sem gerar nenhum erro — só um teste de unicidade pegou. Mostra por que testar o pipeline importa tanto quanto testar o modelo. |
| Como vocês sabem que o modelo não vai degradar em produção? | Monitoramento duplo: PSI e KS nos dados, real e validado com 101 mil clientes, e queda de AUC no desempenho, simulado porque o rótulo real leva 2 anos — os dois testados nos dois sentidos, sem alarme à toa e com alarme quando precisa. |
| O que vocês fariam diferente? | Escrever e congelar o critério de erro de cada hipótese antes de rodar qualquer teste, desde o início — algumas só ganharam esse rigor depois de uma ambiguidade de enunciado aparecer no meio do caminho. |
