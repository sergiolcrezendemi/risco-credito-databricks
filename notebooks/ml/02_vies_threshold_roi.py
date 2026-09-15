# Databricks notebook source
# IMPORTANTE: ESTE COMANDO SÓ E EXECUTADO EM DESENVOLVIMENTO EM PRODUÇÃO SERÁ VIA JOB E A CONFIGURAÇÃO ESTÃO NO ARQUIVO resources/job.yml ou outros arquivo que será executado em produção
%uv sync

# COMMAND ----------

# ==============================================================================
# notebooks/ml/02_vies_threshold_roi.py
# Responde Q1-Q4 do README.md — SHAP, threshold sob custo assimétrico, viés
# por idade/renda e impacto financeiro. Lógica em
# src/models/bias_roi_validation.py; este notebook só orquestra e plota.
#
# Complementar a este: notebooks/ml/01_hipoteses_h1_h4.py (hipóteses H1-H4).
# ==============================================================================

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

MODEL_NAME = "credito_risco_xgb"
MODEL_ALIAS = "champion"

# COMMAND ----------

import sys
import os


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

from src.models import bias_roi_validation as brv
from src.models.hypothesis_validation import load_gold_training_data
import matplotlib.pyplot as plt
import shap

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Dados e modelo

# COMMAND ----------

df = load_gold_training_data(spark, catalog=catalog)
X_train, X_test, y_train, y_test = brv.prepare_train_test(df)

model = brv.load_or_train_model(
    spark, catalog=catalog, schema="gold", model_name=MODEL_NAME, model_alias=MODEL_ALIAS,
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
)
y_proba_test = model.predict_proba(X_test)[:, 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q1 — Importância de variáveis via SHAP

# COMMAND ----------

ranking_shap, shap_values = brv.compute_shap_ranking(model, X_test)
print(ranking_shap.to_string(index=False))

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

top_features = set(ranking_shap.head(5)["feature"])
status_h1 = brv.check_top_n(brv.H1_FEATURES, top_features)
status_h2 = brv.check_top_n(brv.H2_FEATURES, top_features)
print(f"\nH1 (atrasos específicos no top-5): {status_h1}")
print(f"H2 (revolving_utilization/monthly_income no top-5): {status_h2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q2 — Threshold ótimo sob custo assimétrico
# MAGIC Custos vêm de `src/config/business_params.py` — mesma fonte usada em
# MAGIC H4 (`01_hipoteses_h1_h4.py`). O número final ainda pode diferir de H4
# MAGIC porque usa um threshold diferente (aqui, o que minimiza custo; em H4,
# MAGIC o menor que respeita o piso de aprovação) — isso é intencional.

# COMMAND ----------

threshold_result = brv.optimize_threshold_asymmetric(y_test, y_proba_test)
print(f"Threshold ótimo: {threshold_result['threshold_otimo']:.2f}")
print(f"Custo no threshold ótimo: R$ {threshold_result['custo_threshold_otimo']:,.2f}")
print(f"Custo no threshold 0.50: R$ {threshold_result['custo_threshold_050']:,.2f}")
print(f"Economia: R$ {threshold_result['economia']:,.2f}")

df_custos = threshold_result["df_custos"]
plt.figure(figsize=(8, 5))
plt.plot(df_custos["threshold"], df_custos["custo_total"])
plt.axvline(threshold_result["threshold_otimo"], color="red", linestyle="--",
            label=f"Ótimo = {threshold_result['threshold_otimo']:.2f}")
plt.xlabel("Threshold de probabilidade")
plt.ylabel("Custo total esperado (R$)")
plt.title("Custo esperado vs. Threshold (custo assimétrico)")
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q3 — Estabilidade / viés por faixa de idade e renda

# COMMAND ----------

bias_result = brv.bias_analysis(X_test, y_test, y_proba_test, threshold_result["threshold_otimo"])
if bias_result["aviso_renda"]:
    print(f"[AVISO] {bias_result['aviso_renda']}")

print("\n=== Performance por faixa etária ===")
print(bias_result["tabela_idade"].to_string(index=False))
print("\n=== Performance por faixa de renda ===")
print(bias_result["tabela_renda"].to_string(index=False))

if bias_result["alertas"]:
    print("\n[ALERTA] Grupos com FPR/FNR desproporcional (> média + 1 desvio-padrão):")
    for alerta in bias_result["alertas"]:
        print(f"  {alerta}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q4 — Impacto financeiro estimado

# COMMAND ----------

financial_result = brv.financial_impact(bias_result["df_val"])
print(f"Operações analisadas: {financial_result['n_operacoes']}")
print(f"Taxa de inadimplência sem modelo: {financial_result['taxa_sem_modelo']:.2%} "
      f"-> Perda estimada: R$ {financial_result['perda_sem_modelo']:,.2f}")
print(f"Taxa de inadimplência com modelo: {financial_result['taxa_com_modelo']:.2%} "
      f"-> Perda estimada: R$ {financial_result['perda_com_modelo']:,.2f}")
print(f"Redução bruta de inadimplência: R$ {financial_result['reducao_bruta']:,.2f}")
print(f"Custo de oportunidade (bons pagadores negados): R$ {financial_result['custo_oportunidade']:,.2f}")
print(f"Impacto financeiro líquido estimado: R$ {financial_result['impacto_liquido']:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registro no MLflow

# COMMAND ----------

brv.log_validation_to_mlflow(threshold_result, financial_result, ranking_shap, bias_result, status_h1, status_h2)