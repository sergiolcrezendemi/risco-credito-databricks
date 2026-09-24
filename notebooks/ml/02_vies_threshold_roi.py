# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "shap>=0.44",
#   "xgboost>=2.0",
# ]
# ///
# MAGIC %%sh
# MAGIC # IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
# MAGIC uv sync

# COMMAND ----------

# ==============================================================================
# notebooks/ml/02_vies_threshold_roi.py
# Q1-Q4 do README: SHAP, threshold, viés por idade/renda, calibração,
# comparação com regressão logística e impacto financeiro. Lógica em
# src/models/bias_roi_validation.py; este notebook só orquestra e plota.
#
# Complementar a este: notebooks/ml/01_hipoteses_h1_h4.py (hipóteses H1-H4).
# ==============================================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros
# MAGIC `threshold_operacional` é a régua de decisão definida em H3 (menor
# MAGIC inadimplência respeitando o piso de 70% de aprovação). Viés e impacto
# MAGIC financeiro são medidos nela — é a régua que seria usada em produção.

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.text("threshold_operacional", "0.45")
catalog = dbutils.widgets.get("catalog")
THRESHOLD_OPERACIONAL = float(dbutils.widgets.get("threshold_operacional"))

MODEL_NAME = "credito_risco_xgb"
MODEL_ALIAS = "champion"

# COMMAND ----------

import os
import sys


def _find_repo_root(start: str) -> str:
    path = start
    for _ in range(6):
        if os.path.isdir(os.path.join(path, "src")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    raise RuntimeError(f"Não encontrei a pasta 'src' subindo a partir de {start}.")


repo_root = _find_repo_root(os.getcwd())
if repo_root not in sys.path:
    sys.path.append(repo_root)

import matplotlib.pyplot as plt
import shap

from src.models import bias_roi_validation as brv
from src.models.hypothesis_validation import load_gold_training_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dados e modelo
# MAGIC `prepare_train_test` descarta as linhas do lote de scoring (alvo nulo) e usa
# MAGIC as features explícitas de `FEATURE_COLS`. O split (25%, seed 42, estratificado)
# MAGIC é o mesmo do treino do @champion; `verificar_split_do_champion` confirma isso
# MAGIC comparando a AUC recalculada com a registrada no run de treino.

# COMMAND ----------

df = load_gold_training_data(spark, catalog=catalog)
X_train, X_test, y_train, y_test = brv.prepare_train_test(df)
print(f"Treino: {len(X_train):,} | Teste: {len(X_test):,} | Inadimplência no teste: {y_test.mean():.2%}")

model = brv.load_champion_model(catalog, "gold", MODEL_NAME, MODEL_ALIAS)
y_proba_test = model.predict_proba(X_test)[:, 1]

# Garante que o teste é o mesmo do treino do @champion (sem vazamento)
from sklearn.metrics import roc_auc_score

brv.verificar_split_do_champion(
    catalog, roc_auc_score(y_test, y_proba_test), "gold", MODEL_NAME, MODEL_ALIAS
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q1 — Importância de variáveis via SHAP

# COMMAND ----------

ranking_shap, shap_values = brv.compute_shap_ranking(model, X_test)
print(ranking_shap.to_string(index=False))

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q2 — Threshold de custo mínimo vs. threshold operacional
# MAGIC O threshold de custo mínimo usa os custos de `src/config/business_params.py`.
# MAGIC A diferença de custo entre os dois é o preço de manter o piso de aprovação.

# COMMAND ----------

threshold_result = brv.optimize_threshold_asymmetric(y_test, y_proba_test, THRESHOLD_OPERACIONAL)
for rotulo, chave in [("Custo mínimo", "custo_minimo"), ("Operacional (H3)", "operacional")]:
    print(
        f"{rotulo:<18} threshold {threshold_result[f'threshold_{chave}']:.2f} | "
        f"aprovação {threshold_result[f'aprovacao_threshold_{chave}']:.1%} | "
        f"custo R$ {threshold_result[f'custo_threshold_{chave}']:,.0f}"
    )
print(f"Custo do piso de aprovação: R$ {threshold_result['custo_do_piso_de_aprovacao']:,.0f}")

df_custos = threshold_result["df_custos"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_custos["threshold"], df_custos["custo_total"])
ax.axvline(threshold_result["threshold_custo_minimo"], color="gray", linestyle=":",
           label=f"Custo mínimo = {threshold_result['threshold_custo_minimo']:.2f}")
ax.axvline(THRESHOLD_OPERACIONAL, color="red", linestyle="--",
           label=f"Operacional = {THRESHOLD_OPERACIONAL:.2f}")
ax.set_xlabel("Threshold de probabilidade")
ax.set_ylabel("Custo total esperado (R$)")
ax.set_title("Custo esperado vs. threshold")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q3 — Viés por faixa de idade e renda (threshold operacional)
# MAGIC - **razao_aprovacao**: aprovação do grupo ÷ maior aprovação entre os grupos.
# MAGIC   Abaixo de 0,80 (regra dos 4/5) indica impacto desproporcional. Ler junto com
# MAGIC   `inadimplencia_real`: diferença de aprovação explicada por diferença real de
# MAGIC   risco é esperada; o alerta pede investigação, não conclusão.
# MAGIC - **bons_negados**: proporção de bons pagadores do grupo que o modelo recusa.
# MAGIC   É a métrica de justiça central — mede quem paga pelo erro do modelo. O alerta
# MAGIC   só dispara se o intervalo de confiança inteiro ficar acima da média geral.
# MAGIC - Renda não informada é uma faixa própria.

# COMMAND ----------

import pandas as pd

pd.set_option("display.width", 250)
bias_result = brv.bias_analysis(X_test, y_test, y_proba_test, THRESHOLD_OPERACIONAL)

print("=== Por faixa etária ===")
print(bias_result["tabela_idade"].to_string(index=False))
print("\n=== Por faixa de renda ===")
print(bias_result["tabela_renda"].to_string(index=False))

if bias_result["alertas"]:
    print("\n[ALERTAS DE VIÉS]")
    for alerta in bias_result["alertas"]:
        print(f"  {alerta}")
else:
    print("\nNenhum alerta de viés.")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, (nome, tabela) in zip(axes, [("Faixa etária", bias_result["tabela_idade"]),
                                      ("Faixa de renda", bias_result["tabela_renda"])]):
    ax.bar(tabela["grupo"], tabela["bons_negados"], color="#1f4e79")
    ax.axhline(bias_result["df_val"].query("y_true == 0")["y_pred"].mean(), color="red",
               linestyle="--", label="Média geral")
    ax.set_title(nome)
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("Bons pagadores negados")
axes[0].legend()
plt.suptitle(f"Bons pagadores negados por grupo (threshold {THRESHOLD_OPERACIONAL:.2f})")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q3b — Calibração das probabilidades
# MAGIC Para precificar (taxa de juros por risco), a probabilidade prevista precisa
# MAGIC corresponder à inadimplência observada. Modelos treinados com
# MAGIC `scale_pos_weight` costumam superestimar o risco. A calibração isotônica é
# MAGIC ajustada em metade do teste e avaliada na outra metade; ela não muda a
# MAGIC ordenação dos clientes, só o valor da probabilidade.

# COMMAND ----------

calibration_result = brv.calibration_analysis(y_test, y_proba_test)
print(f"Inadimplência observada: {calibration_result['inadimplencia_observada']:.2%}")
for versao in ["bruta", "calibrada"]:
    r = calibration_result[versao]
    print(f"{versao:<10} média prevista {r['media_prevista']:.2%} | Brier {r['brier']:.4f} | "
          f"erro médio de calibração {r['erro_calibracao_medio']:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Calibração perfeita")
for versao, cor in [("bruta", "#c0392b"), ("calibrada", "#1f4e79")]:
    curva = calibration_result[versao]["curva"]
    ax.plot(curva["prob_prevista"], curva["inadimplencia_observada"], marker="o", color=cor,
            label=versao.capitalize())
ax.set_xlabel("Probabilidade prevista")
ax.set_ylabel("Inadimplência observada")
ax.set_title("Curva de calibração")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q3c — XGBoost vs. regressão logística na mesma taxa de aprovação
# MAGIC Compara as carteiras aprovando a mesma proporção de clientes do threshold
# MAGIC operacional — quanto o XGBoost agrega sobre um modelo mais simples.

# COMMAND ----------

comparacao = brv.compare_with_logistic(
    X_train, y_train, X_test, y_test, y_proba_test,
    taxa_aprovacao=threshold_result["aprovacao_threshold_operacional"],
)
print(f"Taxa de aprovação comum: {threshold_result['aprovacao_threshold_operacional']:.1%}\n")
print(comparacao.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q3d — Experimento: o mesmo modelo sem a variável idade
# MAGIC Treina um XGBoost com os mesmos hiperparâmetros do @champion, sem `age`, e
# MAGIC compara os dois na mesma taxa de aprovação. Mede quanto do viés etário vem da
# MAGIC variável em si e quanto chega por outras variáveis que carregam informação de
# MAGIC idade (proxies), e quanto desempenho se perde.

# COMMAND ----------

experimento_sem_idade = brv.experimento_sem_variavel(
    X_train, y_train, X_test, y_test, y_proba_test,
    taxa_aprovacao=threshold_result["aprovacao_threshold_operacional"],
)
print(experimento_sem_idade["resumo"].to_string(index=False))
print("\n=== Bons pagadores negados por faixa etária (mesma taxa de aprovação) ===")
print(experimento_sem_idade["bons_negados_por_faixa"].to_string(index=False))

tabela = experimento_sem_idade["bons_negados_por_faixa"]
modelos_exp = [c for c in tabela.columns if c not in ("faixa_idade", "variacao_pp")]
fig, ax = plt.subplots(figsize=(9, 4.5))
largura = 0.38
posicoes = range(len(tabela))
for deslocamento, nome, cor in zip([-largura / 2, largura / 2], modelos_exp, ["#1f4e79", "#e67e22"]):
    ax.bar([p + deslocamento for p in posicoes], tabela[nome], width=largura, label=nome, color=cor)
ax.set_xticks(list(posicoes))
ax.set_xticklabels(tabela["faixa_idade"])
ax.set_ylabel("Bons pagadores negados")
ax.set_title("Bons pagadores negados por faixa etária: com e sem a variável idade")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q4 — Impacto financeiro (threshold operacional)

# COMMAND ----------

financial_result = brv.financial_impact(bias_result["df_val"])
print(f"Operações analisadas: {financial_result['n_operacoes']:,} | aprovação {financial_result['taxa_aprovacao']:.1%}")
print(f"Inadimplência: {financial_result['taxa_sem_modelo']:.2%} sem modelo -> "
      f"{financial_result['taxa_com_modelo']:.2%} com modelo")
print(f"Perda evitada ({financial_result['maus_recusados']:,} maus recusados): "
      f"R$ {financial_result['perda_evitada']:,.0f}")
print(f"Custo de oportunidade ({financial_result['bons_negados']:,} bons negados): "
      f"R$ {financial_result['custo_oportunidade']:,.0f}")
print(f"Impacto líquido: R$ {financial_result['impacto_liquido']:,.0f}")
print(f"Ponto de equilíbrio: o modelo se paga enquanto a perda por calote for maior que "
      f"{financial_result['razao_equilibrio_perda_margem']:.1f}x a margem de um bom cliente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registro no MLflow

# COMMAND ----------

brv.log_validation_to_mlflow(
    threshold_result, financial_result, ranking_shap, bias_result, calibration_result, comparacao,
    experimento_sem_idade,
)
