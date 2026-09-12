# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///


# COMMAND ----------

# ==============================================================================
# notebooks/ml/01_hipoteses_h1_h4.py
# Validação das hipóteses H1-H4 (ver README.md) — treino, SHAP e teste de
# cada hipótese vivem em src/models/hypothesis_validation.py. Este notebook
# só orquestra e plota; nenhuma lógica de negócio deve ser adicionada aqui.
#
# Complementar a este: notebooks/ml/02_vies_threshold_roi.py (perguntas
# Q2/Q3 do README — threshold com custo assimétrico e checagem de viés por
# idade/renda), que já lê da mesma Gold.
# ==============================================================================

# COMMAND ----------

dbutils.widgets.text("catalog", "credito_dev")
catalog = dbutils.widgets.get("catalog")

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

# %pip install -r requirements.txt -qqq  (rode manualmente se faltar xgboost/shap)

from src.models import hypothesis_validation as hv
import matplotlib.pyplot as plt

COR_CONFIRMADA = "#2E7D32"
COR_ALERTA = "#C62828"
COR_NEUTRA = "#1565C0"
COR_BASELINE = "#757575"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Treino (XGBoost + Regressão Logística) e SHAP

# COMMAND ----------

models = hv.run_training(spark, catalog=catalog)
shap_sample, shap_values, shap_importance_df = hv.compute_shap(models)
logreg_coefs = hv.get_logreg_coefficients(models)

print("\n--- Ranking de Importância SHAP (Top Features) ---")
print(shap_importance_df)

fig, ax = plt.subplots(figsize=(9, 5))
plot_df = shap_importance_df.sort_values("mean_abs_shap", ascending=True)
ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color=COR_NEUTRA)
ax.set_xlabel("Impacto médio no modelo (|SHAP|)")
ax.set_title("Importância das Features (SHAP) — Ranking Geral")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## H1 — Utilização de crédito rotativo

# COMMAND ----------

h1 = hv.validate_h1(shap_sample, shap_values, shap_importance_df)
print(f"Variável: {h1['feature']}")
print(f"Posição no ranking SHAP: {h1['posicao_ranking']}º (|SHAP|: {h1['mean_abs_shap']:.4f})")
print(f"Correlação Feature vs SHAP: {h1['correlacao']:.4f}")
print(f"Conclusão H1: {h1['status']}")

cor_h1 = COR_CONFIRMADA if h1["status"] == "Confirmada" else COR_ALERTA
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(shap_sample[h1["feature"]], shap_values[:, h1["feature"]].values, alpha=0.35, s=12, color=cor_h1)
ax.axhline(0, color=COR_BASELINE, linewidth=1, linestyle="--")
ax.set_xlim(-0.05, 2.0)
ax.set_xlabel(h1["feature"])
ax.set_ylabel("Valor SHAP (impacto na previsão)")
ax.set_title(f"H1 — {h1['feature']} vs. SHAP | correlação = {h1['correlacao']:.3f} | {h1['status']}")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## H2 — Dívida/renda como preditor dominante

# COMMAND ----------

h2 = hv.validate_h2(shap_importance_df)
print(f"debt_ratio -> |SHAP|: {h2['shap_a']:.4f} | age -> |SHAP|: {h2['shap_b']:.4f}")
print(f"Diferença relativa: {h2['diferenca_relativa']:.2%}")
print(f"Conclusão H2: {h2['status']}")

cor_h2 = COR_CONFIRMADA if h2["status"] == "Confirmada" else COR_ALERTA
fig, ax = plt.subplots(figsize=(6, 5))
barras = ax.bar([h2["feature_a"], h2["feature_b"]], [h2["shap_a"], h2["shap_b"]], color=[cor_h2, COR_BASELINE])
for b in barras:
    altura = b.get_height()
    ax.annotate(f"{altura:.4f}", (b.get_x() + b.get_width() / 2, altura), textcoords="offset points",
                xytext=(0, 4), ha="center")
ax.set_ylabel("Impacto médio no modelo (|SHAP|)")
ax.set_title(f"H2 — debt_ratio vs age | diferença relativa = {h2['diferenca_relativa']:.1%} | {h2['status']}")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## H3 — Threshold ótimo de aprovação

# COMMAND ----------

h3 = hv.validate_h3(models.y_test, models.y_pred_xgb)
print(f"Baseline (sem filtro): {h3['baseline_default_rate']:.2%} | Piso de aprovação: {h3['min_approval_rate']:.0%}")
if h3["best_row"] is not None:
    br = h3["best_row"]
    print(f"Melhor threshold: {br['threshold']} | Aprovação: {br['approval_rate']:.2%} | "
          f"Inadimplência aprovados: {br['default_rate_approved']:.2%}")
print(f"Conclusão H3: {h3['status']}")

threshold_df = h3["threshold_df"]
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(threshold_df["threshold"], threshold_df["approval_rate"], marker="o", color=COR_NEUTRA, label="Taxa de aprovação")
ax1.axhline(h3["min_approval_rate"], color=COR_BASELINE, linestyle="--", label="Piso de aprovação")
ax1.set_xlabel("Threshold de probabilidade")
ax1.set_ylabel("Taxa de aprovação", color=COR_NEUTRA)
ax2 = ax1.twinx()
ax2.plot(threshold_df["threshold"], threshold_df["default_rate_approved"], marker="s", color=COR_ALERTA,
         label="Inadimplência na carteira aprovada")
ax2.axhline(h3["baseline_default_rate"], color=COR_ALERTA, linestyle=":", label="Baseline sem filtro")
ax2.set_ylabel("Inadimplência aprovados", color=COR_ALERTA)
if h3["best_row"] is not None:
    ax1.axvline(h3["best_row"]["threshold"], color=COR_CONFIRMADA, linewidth=1.5, label="Melhor threshold")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="center left", fontsize=8)
ax1.set_title(f"H3 — Aprovação vs. Inadimplência por Threshold | {h3['status']}")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## H4 — Impacto financeiro vs. política atual
# MAGIC `PERDA_MEDIA_POR_INADIMPLENCIA_R` / `MARGEM_MEDIA_POR_CLIENTE_BOM_R`
# MAGIC vêm de `src/config/business_params.py` — mesma fonte do notebook de
# MAGIC viés/ROI (`02_vies_threshold_roi.py`). O valor final ainda pode
# MAGIC diferir de Q4 porque usa o threshold de H3 (piso de aprovação), não
# MAGIC o de custo mínimo — isso é intencional.

# COMMAND ----------

h4 = hv.validate_h4(models.y_test, models.y_pred_xgb, h3)
if h4["impacto_liquido"] is not None:
    print(f"Threshold usado (herdado de H3): {h4['threshold_usado']}")
    print(f"Maus pagadores evitados: {h4['maus_pagadores_evitados']} -> Perda evitada: R$ {h4['perda_evitada']:,.2f}")
    print(f"Bons pagadores rejeitados: {h4['bons_pagadores_rejeitados']} -> Custo de oportunidade: R$ {h4['custo_oportunidade']:,.2f}")
    print(f"Impacto financeiro líquido: R$ {h4['impacto_liquido']:,.2f}")
    print(f"Conclusão H4: {h4['status']}")

    cor_liquido = COR_CONFIRMADA if h4["impacto_liquido"] > 0 else COR_ALERTA
    labels = ["Perda evitada", "Custo de oportunidade", "Impacto líquido"]
    valores = [h4["perda_evitada"], -h4["custo_oportunidade"], h4["impacto_liquido"]]
    cores = [COR_CONFIRMADA, COR_ALERTA, cor_liquido]
    fig, ax = plt.subplots(figsize=(7, 5))
    barras = ax.bar(labels, valores, color=cores)
    ax.axhline(0, color=COR_BASELINE, linewidth=1)
    for b in barras:
        altura = b.get_height()
        ax.annotate(f"R$ {altura:,.0f}", (b.get_x() + b.get_width() / 2, altura), textcoords="offset points",
                    xytext=(0, 6 if altura >= 0 else -14), ha="center", fontsize=8)
    ax.set_ylabel("R$ (valores ilustrativos)")
    ax.set_title(f"H4 — Impacto Financeiro Líquido | {h4['status']}")
    plt.tight_layout()
    plt.show()
else:
    print(f"Conclusão H4: {h4['status']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persistência do resumo das hipóteses na Gold

# COMMAND ----------

summary_df = hv.build_hypotheses_summary(h1, h2, h3, h4)
display(spark.createDataFrame(summary_df))
hv.persist_hypotheses_summary(spark, summary_df, catalog=catalog)