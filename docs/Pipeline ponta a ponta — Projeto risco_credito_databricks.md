# Pipeline ponta a ponta — Projeto risco_credito_databricks

## Visão geral do fluxo

```
Bronze (ingestão)
      │  Autoloader + Unity Catalog
      ▼
Silver / Gold (preparação)
      │  Limpeza e features
      ▼
Feature Store
      │  Features versionadas
      ▼
Treinamento (MLflow)
      │  Tracking de experimentos
      ▼
Registro + Deploy
      │  Model Registry, serving
      ▼
Monitoramento
      │  Drift e alertas
      ▼
   ↻ Drift detectado aciona retreino, voltando ao Bronze
```

## Como o cientista de dados deveria pensar em cada etapa

### 1. Bronze (ingestão)
A pergunta aqui não é "como eu leio o CSV", é "como esse dado vai chegar de forma contínua e confiável quando isso estiver em produção". Por isso o uso do Autoloader com Unity Catalog: ele já resolve schema evolution e ingestão incremental, então não é preciso reprocessar a base inteira toda vez que chega um lote novo.

### 2. Silver / Gold (preparação)
O raciocínio muda de "dado bruto" para "dado que faz sentido de negócio". É a etapa de limpar, tratar outliers, tipar corretamente, e já pensar em quais transformações vão virar *features* — ou seja, decisões de engenharia de dados que antecipam decisões de modelagem.

### 3. Feature Store
A pergunta central é: "essa feature vai ser recalculada do mesmo jeito em treino e em produção?". Sem isso há risco de *training-serving skew*: o modelo aprende com uma versão da feature e é servido com outra ligeiramente diferente. Versionar as features também dá rastreabilidade — essencial num contexto de crédito, que costuma ter exigência de auditoria.

### 4. Treinamento (MLflow)
Aqui o cientista de dados para de pensar só em "qual modelo dá a melhor métrica" e passa a pensar em reprodutibilidade: cada experimento precisa registrar hiperparâmetros, métricas e a versão exata dos dados usados, porque alguém (inclusive o próprio autor, meses depois) vai precisar explicar por que aquele modelo foi escolhido.

### 5. Registro + Deploy
A pergunta muda de "o modelo funciona" para "o modelo está pronto para ser consumido por um sistema real, com contrato de entrada/saída estável e um processo formal de promoção (staging → produção)".

### 6. Monitoramento
O raciocínio é: "o mundo muda depois que o modelo vai para produção". Drift de dados ou de performance é o sinal de que o modelo aprendeu um padrão que já não reflete a realidade — e é isso que fecha o ciclo, disparando um retreino.

---

Esse fio condutor — de "dado bruto sem confiabilidade" até "modelo em produção sendo continuamente vigiado" — é o que separa um notebook de experimentação de um pipeline de nível sênior.
