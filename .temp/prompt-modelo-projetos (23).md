# Prompt-modelo para novos projetos

Rascunho para colar como primeira mensagem do próximo projeto, ajustando o que está entre colchetes.

O prompt tem duas partes: um **núcleo obrigatório**, que vale para
qualquer projeto, e um **checklist estendido**, que só se aplica
quando o horizonte do projeto for "produção real" (não em POC/
exploração rápida). Isso evita que uma investigação de poucas horas
fique travada em exigências de infraestrutura de produção.

Para projetos de "produção real", enviar em duas mensagens em vez de
uma só, para reduzir o risco de itens do meio/final de uma lista
longa receberem menos atenção que os do início:
1. Primeira mensagem: bloco de contexto + núcleo obrigatório. Pedir
   confirmação do entendimento em formato de checklist antes de
   prosseguir.
2. Segunda mensagem (após a confirmação): checklist estendido, mais
   qualquer material já pronto (dicionário de dados, base de dados
   etc.).

Para POC rápida, o núcleo obrigatório sozinho, numa mensagem, já
basta.

```
Quero construir [nome do projeto], que sustenta a seguinte decisão de
negócio: [decisão em uma frase — o que alguém decide, com base nisso].

Prazo/horizonte do projeto: [POC rápida para validar hipótese / projeto
para produção real]. Isso define se o checklist estendido abaixo se
aplica ou não.

Escopo: entra [o que entra]. Fica de fora, por enquanto: [o que fica
para depois]. Métrica técnica alvo: [ex. AUC ≥ 0,85 para classificação,
ou silhouette score / validação de negócio para agrupamento]. Métrica
de negócio: [ex. redução de X% na inadimplência] — se o número real
ainda não existe, marque como "ilustrativo, a validar", nunca como
definitivo.

Consumidor final do resultado: [dashboard / API / score em batch /
relatório / apresentação] — isso define boa parte da arquitetura,
então precisa estar claro desde já.

Ambiente de execução: [local / Docker com a imagem poc-datascience /
Databricks / nuvem]. Caminho de promoção: [dev → staging → produção,
ou "só POC local por enquanto"].

Volume de dados / restrição computacional: [a base cabe em memória
(pandas) ou precisa de processamento distribuído (Spark) desde o
início?] — isso muda a abordagem técnica logo na primeira linha de
código.

Dados sensíveis: [há dado pessoal/de saúde/financeiro envolvido? Se
sim, sinalizar tratamento de PII e conformidade LGPD antes de
qualquer modelagem; se não, declarar isso também].

=== NÚCLEO OBRIGATÓRIO (sempre, mesmo em POC) ===

Antes de qualquer código, quero fechar:
1. As hipóteses, cada uma com palpite + teste + critério de erro +
   decisão que muda, escritas e congeladas antes do primeiro teste.
2. Configuração centralizada: parâmetros reutilizados e não sensíveis
   (ambiente, caminhos, nomes de tabelas/catálogos, thresholds,
   limiares) em arquivo .yaml — nunca duplicado em módulos
   diferentes; os módulos devem referenciar esse arquivo, não
   redeclarar valores. Credenciais, senhas e tokens vão em .env
   (nunca versionado — sempre no .gitignore), carregado via biblioteca
   apropriada (ex. python-dotenv); em produção no Databricks, avaliar
   uso de Secrets scope no lugar do .env.
3. Riscos e limitações conhecidos, nomeados agora — [ex. base pública
   como proxy, viés de seleção, ausência de variável temporal].
4. Critério de "pronto" para cada fase do projeto — o que precisa ser
   verdade para considerar aquela fase fechada e não reabrir depois.
5. Toda métrica, número ou resultado de teste apresentado deve vir de
   código efetivamente executado — nunca estimado ou descrito como se
   fosse resultado real. Marcar explicitamente qualquer afirmação que
   ainda não foi verificada por execução (ex. "[não executado — a
   confirmar]"), em vez de apresentá-la como fato. Para uso de
   biblioteca/API pouco comum, confirmar a sintaxe antes de assumir
   que está correta.

Diretrizes de arquitetura e código (sempre):
- Estrutura de diretórios: sempre a mesma, definida no script de
  criação de projeto (notebooks/, src/<pacote>/, tests/, resources/,
  docs/, entre outras pastas padrão). Isso é premissa fixa, não uma
  decisão a ser tomada por projeto — não criar estrutura própria nem
  variar entre projetos.
- Resolução de caminhos entre arquivos deve usar uma referência fixa
  de raiz do projeto (ex. uma constante PROJECT_ROOT resolvida via
  pathlib, não sobe/desce diretório contando "..", "../.." manualmente
  em cada script) — isso evita erros recorrentes de caminho quebrado
  quando um módulo referencia outro em profundidade diferente na
  árvore de pastas.
- Código modular e reutilizável desde a v1 — nenhuma função duplicada
  entre notebooks/scripts. Funções devem ficar em arquivo(s)
  separado(s) (ex. utils.py, features.py), e o notebook deve apenas
  importar e chamar essas funções — nada de lógica de negócio
  implementada diretamente nas células.
- Dependências fixadas em pyproject.toml (não requirements.txt) para
  reprodutibilidade do ambiente e evitar conflitos de pacote.
- Para cada escolha técnica relevante (algoritmo, validação, técnica
  de tratamento de desbalanceamento etc.), justificar por que essa
  opção foi escolhida em vez de alternativas comuns — evitar "melhores
  práticas" genéricas sem justificativa concreta para o caso.

Convenções adicionais (sempre):
- Contrato de dados de entrada: schema esperado das tabelas de
  origem (nomes de colunas, tipos, chave primária) declarado antes
  de começar a modelagem.
- Ao combinar mais de uma fonte/arquivo com IDs próprios (ex. treino
  + scoring, ou bases de sistemas diferentes), verificar
  explicitamente se as faixas de ID podem se sobrepor ANTES de unir
  as fontes — não confiar apenas em teste de unicidade rodado depois,
  já que uma colisão de ID pode misturar registros de populações
  diferentes sem gerar erro (ex. troca de atributos entre um cliente
  de treino e um de scoring com o mesmo ID). Se houver risco de
  sobreposição, aplicar deslocamento ou chave composta antes da
  união.
- Estratégia de split treino/teste/validação declarada — quando
  houver componente temporal, usar split temporal (não aleatório),
  para não vazar informação do futuro no treino.
- Seed fixo em todas as etapas com componente aleatório, para
  garantir que os resultados possam ser recalculados de forma
  idêntica.
- Convenção de nomenclatura única para tabelas, colunas, experimentos
  MLflow etc., evitando inconsistência entre módulos criados em
  momentos diferentes.
- Convenção de idioma: código e nomes técnicos em inglês,
  documentação e comentários em português — evitar mistura
  inconsistente entre os dois.

Formato de trabalho: construir de forma incremental — um
módulo/notebook por vez, revisado antes de seguir para o próximo —
em vez de entregar tudo de uma vez. Antes de iniciar cada novo
módulo, reler os arquivos de configuração, contrato de dados e
dicionário de dados já existentes no projeto, em vez de assumir que
o contexto da conversa é suficiente — a fonte da verdade é o
arquivo, não a memória da conversa, especialmente em projetos longos
com muitas sessões.

Identificação: toda conversa sobre este projeto deve incluir a tag
#TAG_PROJETOx (substituir "x" pelo identificador do projeto), para
manter o contexto rastreável entre conversas.

Ajustes conforme o tipo de problema — [classificação / agrupamento]:
- Métrica técnica alvo: em classificação, usar métrica supervisionada
  (ex. AUC, F1, precisão/recall). Em agrupamento, usar métrica de
  qualidade de cluster (ex. silhouette score, Davies-Bouldin) e/ou
  validação de que os grupos fazem sentido para o negócio — não há
  variável alvo para comparar.
- Split treino/teste/validação: em classificação, usar split (temporal
  quando aplicável) para evitar vazamento de informação futura. Em
  agrupamento, não há "vazamento" no mesmo sentido — usar holdout para
  testar estabilidade dos clusters (os mesmos grupos aparecem numa
  amostra diferente da base?).

Se o horizonte for "produção real": antes de eu ver o checklist
estendido, quero que você devolva um checklist confirmando seu
entendimento de cada ponto do núcleo obrigatório acima (decisão de
negócio, escopo, métricas, hipóteses, configuração, estrutura,
contrato de dados etc.). Só depois dessa confirmação eu envio o
checklist estendido e qualquer material já pronto (dicionário de
dados, base de dados).

=== CHECKLIST ESTENDIDO (só se o horizonte for "produção real") ===

- Se o projeto envolver pipeline de dados com múltiplas transformações,
  usar arquitetura medalhão (bronze/silver/gold).
- Ingestão incremental (padrão Autoloader) na camada bronze quando o
  projeto envolver dados que chegam ao longo do tempo — evitar
  reprocessar a base inteira a cada execução. Garantir que schema e
  volumes/diretórios de destino (incluindo o de checkpoint) já
  existam antes de iniciar o writeStream — referenciar um checkpoint
  antes do volume existir quebra o pipeline com erro de infraestrutura
  não encontrada.
- Testes automáticos: unicidade de chave em qualquer união de
  tabelas, reprodutibilidade do modelo registrado (recalcular e
  comparar a métrica antes de confiar nela), e validação de qualquer
  teste estatístico contra a distribuição real da variável. Quando
  houver alias de produção no Model Registry (ex. @champion), travar
  a validação: a métrica recalculada no teste precisa bater
  exatamente com a métrica registrada no run desse modelo — se não
  bater, a execução deve parar, em vez de seguir com números que
  parecem coerentes mas vieram de versões diferentes do modelo.
- Usar MLflow para tracking de experimentos e Model Registry —
  todo treino deve logar parâmetros, métricas e artefatos, e o
  modelo deve ser registrado formalmente com estágios explícitos
  (ex. staging → produção) e versão rastreável — nunca ficar apenas
  em arquivo solto.
- Ao carregar o modelo registrado para inferência, usar o flavor
  nativo do modelo (ex. XGBoost, sklearn), não o flavor genérico
  pyfunc — carregar via pyfunc pode devolver a classe prevista (0/1)
  em vez da probabilidade, dependendo da versão instalada, causando
  erro silencioso de calibração sem gerar exceção.
- Monitoramento pós-deploy como requisito desde o design: como
  saberemos se o modelo degradou (drift de dados, drift de conceito,
  queda na métrica de negócio)? O ambiente já deve nascer preparado
  para isso — capturar e armazenar as distribuições de referência
  (baseline) das features e da variável alvo desde o início.
- Feature Store centralizado quando houver features reutilizadas
  entre treino e inferência, para evitar divergência entre os dois
  (training-serving skew).
- Governança e linhagem de dados (catálogo, controle de acesso)
  quando o projeto envolver dados sensíveis (saúde, financeiro,
  seguros) — quem pode acessar o quê, e de onde cada tabela vem.
- Dicionário de dados mantido em arquivo separado (ex. docs/
  dicionario_dados.md), documentando cada coluna relevante: nome,
  tipo, significado de negócio e origem — atualizado conforme o
  dado evolui entre as camadas bronze/silver/gold.
- Versionamento de dados: quando usar Delta Lake, fixar
  versão/snapshot da tabela usada no treino (time travel), para que
  o resultado seja rastreável mesmo se a tabela de origem mudar
  depois.
- Responsável pela validação do resultado antes de ir para produção
  identificado desde o início, para não reabrir uma decisão já
  aprovada.
- Explicabilidade do modelo (ex. SHAP) incluída desde o design,
  especialmente em crédito/seguros, onde é comum precisar justificar
  por que um score foi dado. Em agrupamento, usar caracterização/
  perfil de cada cluster no lugar de SHAP.
- Checagem de viés/fairness em atributos sensíveis, validada antes de
  ir para produção.
- Logging estruturado (biblioteca de logging com níveis
  info/warning/error) em vez de print(), para rastrear execução em
  produção.
- Padronização de código via linter/formatter (ex. black, ruff) e
  type hints, aplicados via pre-commit hooks.
- CI/CD rodando os testes automatizados do projeto a cada push, não
  apenas localmente antes de subir código.
- Plano de rollback definido: como reverter rapidamente para a versão
  anterior do modelo em produção caso a nova cause problema.
- Ao final, além do notebook/código técnico, preparar um resumo em
  linguagem de negócio: o que foi decidido, o que mudou, e o que
  fazer com o resultado — não só documentação técnica.

Só depois de fechar o núcleo obrigatório (e o checklist estendido,
se aplicável), quero começar a arquitetura e o código.
```
