# Prompt-modelo para novos projetos

Rascunho para colar como primeira mensagem do próximo projeto, ajustando o que está entre colchetes.

```
Quero construir [nome do projeto], que sustenta a seguinte decisão de
negócio: [decisão em uma frase — o que alguém decide, com base nisso].

Escopo: entra [o que entra]. Fica de fora, por enquanto: [o que fica
para depois]. Métrica técnica alvo: [ex. AUC ≥ 0,85]. Métrica de
negócio: [ex. redução de X% na inadimplência] — se o número real ainda
não existe, marque como "ilustrativo, a validar", nunca como
definitivo.

Consumidor final do resultado: [dashboard / API / score em batch /
relatório / apresentação] — isso define boa parte da arquitetura,
então precisa estar claro desde já.

Ambiente de execução: [local / Docker com a imagem poc-datascience /
Databricks / nuvem]. Caminho de promoção: [dev → staging → produção,
ou "só POC local por enquanto"].

Dados sensíveis: [há dado pessoal/de saúde/financeiro envolvido? Se
sim, sinalizar tratamento de PII e conformidade LGPD antes de
qualquer modelagem; se não, declarar isso também].

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
5. Critério de "pronto" para cada fase do projeto — o que precisa ser
   verdade para considerar aquela fase fechada e não reabrir depois.

Diretrizes de arquitetura e código:
- Se o projeto envolver pipeline de dados com múltiplas transformações,
  usar arquitetura medalhão (bronze/silver/gold). Não aplicar isso
  como regra fixa em POCs simples de camada única.
- Código modular e reutilizável desde a v1 — nenhuma função duplicada
  entre notebooks/scripts. Pensar desde o início em como isso vai
  rodar em produção, não só localmente.
- Para cada escolha técnica relevante (algoritmo, validação, técnica
  de tratamento de desbalanceamento etc.), justificar por que essa
  opção foi escolhida em vez de alternativas comuns — evitar "melhores
  práticas" genéricas sem justificativa concreta para o caso.
- Monitoramento pós-deploy como requisito desde o design, não como
  item posterior: como saberemos se o modelo degradou (drift de
  dados, drift de conceito, queda na métrica de negócio)?

Formato de trabalho: construir de forma incremental — um
módulo/notebook por vez, revisado antes de seguir para o próximo —
em vez de entregar tudo de uma vez.

Só depois de fechar tudo isso, quero começar a arquitetura e o código.
```
