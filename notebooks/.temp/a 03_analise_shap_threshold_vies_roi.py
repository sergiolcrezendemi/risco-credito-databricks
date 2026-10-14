%python
# Databricks notebook source
# =============================================================================
# 03_analise_shap_threshold_vies_roi.py
# -----------------------------------------------------------------------------
# Fase de Validação do case "Risco de Crédito" (Give Me Some Credit + BACEN SCR.data)
#
# Responde às 4 perguntas de negócio do README:
#   Q1 - Importância de variáveis via SHAP (confirma/contraria H1 e H2?)
#   Q2 - Threshold ótimo considerando custo assimétrico (FN vs FP)
#   Q3 - Estabilidade do modelo por faixa de renda e idade (checagem de viés)
#   Q4 - Impacto financeiro estimado (R$) do modelo vs. política atual
#
# Pré-requisitos:
#   - Tabela Gold com features + target já materializada (Unity Catalog)
#   - Um modelo já treinado e registrado no MLflow Model Registry
#     (se ainda não existir, o script treina um XGBoost baseline e registra)
# =============================================================================

# COMMAND ----------
# =========================
# 0. CONFIGURAÇÃO
# =========================
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# --- Ajuste estes parâmetros ao seu ambiente ---
CATALOG = "credito_prd"                        # catálogo real (Catalog Explorer)
SCHEMA = "gold"
FCT_TABLE = "fct_credit_profile"               # tabela fato: métricas de crédito
DIM_CUSTOMER_TABLE = "dim_customer"            # tabela dimensão: idade, dependentes
TARGET_COL = "target_dlq_2yrs"                 # 1 = inadimplente, 0 = bom pagador
MODEL_NAME = "credito_risco_xgb"               # nome no Model Registry
MODEL_STAGE = "Production"                     # ou "Staging" / versão específica

# --- Hipóteses do README (preencha com o texto exato das suas hipóteses) ---
H1_TEXT = "H1: variáveis de atraso histórico (30-59/60-89/90+ dias) são as mais preditivas."
H2_TEXT = "H2: utilização de crédito rotativo (RevolvingUtilization) supera renda como preditor."
H1_FEATURES = [
    "num_times_30_59_days_late",
    "num_times_90_days_late",
]
H2_FEATURES = ["revolving_utilization", "monthly_income"]

# --- Parâmetros de custo assimétrico (Q2) — ajuste aos valores reais do produto ---
CUSTO_APROVAR_MAU_PAGADOR = 15000.0   # custo esperado de conceder crédito a um mau pagador (perda média)
CUSTO_NEGAR_BOM_PAGADOR = 1500.0      # custo de oportunidade de negar crédito a um bom pagador (margem perdida)

# --- Parâmetro de impacto financeiro (Q4) ---
VALOR_MEDIO_OPERACAO = 12000.0        # ticket médio de uma operação de crédito, em R$
TAXA_INADIMPLENCIA_POLITICA_ATUAL = None  # se None, é calculada a partir da base (aprovação de todos)

# COMMAND ----------
# =========================
# 1. CARGA DOS DADOS
# =========================
df_spark = spark.sql(f"""
    SELECT
        f.customer_id,
        c.age,
        c.num_dependents,
        f.monthly_income,
        f.debt_ratio,
        f.revolving_utilization,
        f.num_open_credit_lines,
        f.num_real_estate_loans,
        f.num_times_30_59_days_late,
        f.num_times_60_89_days_late,
        f.num_times_90_days_late,
        f.total_delinquency_events,
        f.{TARGET_COL}
    FROM {CATALOG}.{SCHEMA}.{FCT_TABLE} f
    JOIN {CATALOG}.{SCHEMA}.{DIM_CUSTOMER_TABLE} c
      ON f.customer_id = c.customer_id
""")
df = df_spark.toPandas()

# Remove registros com target nulo (train_test_split/stratify não aceita NaN em y)
n_antes = len(df)
df = df.dropna(subset=[TARGET_COL])
n_removidos = n_antes - len(df)
if n_removidos > 0:
    print(f"[AVISO] {n_removidos} de {n_antes} registros removidos por '{TARGET_COL}' nulo "
          f"({n_removidos / n_antes:.2%}).")
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Colunas que não entram como feature (identificador e target)
ID_COLS = ["customer_id"]
feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]
X = df[feature_cols]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# COMMAND ----------
# =========================
# 2. CARGA (OU TREINO BASELINE) DO MODELO
# =========================
model = None
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.xgboost.load_model(model_uri)
    print(f"Modelo carregado do Registry: {model_uri}")
except Exception as e:
    print(f"Não foi possível carregar do Registry ({e}). Treinando baseline XGBoost...")
    with mlflow.start_run(run_name="baseline_xgb_validacao"):
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # desbalanceamento
            random_state=42,
        )
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("auc_test", auc)
        mlflow.xgboost.log_model(model, "model")
        print(f"AUC baseline no teste: {auc:.4f}")

y_proba_test = model.predict_proba(X_test)[:, 1]

# COMMAND ----------
# =========================
# Q1. IMPORTÂNCIA DE VARIÁVEIS VIA SHAP
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Importância global média |SHAP|
mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking_shap = (
    pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
ranking_shap["rank"] = ranking_shap.index + 1

print("\n=== Q1 — Ranking de importância (SHAP) ===")
print(ranking_shap.to_string(index=False))

# Plots (salvos como artefatos MLflow / arquivos)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("/tmp/shap_summary.png", dpi=150)
plt.close()

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("/tmp/shap_bar.png", dpi=150)
plt.close()

# Checagem automática das hipóteses: a feature está no top-N do ranking?
top_n = 5
top_features = set(ranking_shap.head(top_n)["feature"])


def checa_hipotese(nome, texto, features_esperadas, top_features):
    presentes = [f for f in features_esperadas if f in top_features]
    ausentes = [f for f in features_esperadas if f not in top_features]
    status = "CONFIRMADA" if len(presentes) == len(features_esperadas) else (
        "PARCIALMENTE CONFIRMADA" if presentes else "CONTRARIADA"
    )
    print(f"\n{nome}: {texto}")
    print(f"  Status: {status}")
    print(f"  Presentes no top-{top_n}: {presentes}")
    if ausentes:
        print(f"  Ausentes no top-{top_n}: {ausentes}")
    return status


status_h1 = checa_hipotese("H1", H1_TEXT, H1_FEATURES, top_features)
status_h2 = checa_hipotese("H2", H2_TEXT, H2_FEATURES, top_features)

# COMMAND ----------
# =========================
# Q2. THRESHOLD ÓTIMO SOB CUSTO ASSIMÉTRICO
# =========================
# Erro Tipo I  (negar crédito a bom pagador) = Falso Positivo -> custo CUSTO_NEGAR_BOM_PAGADOR
# Erro Tipo II (aprovar crédito a mau pagador) = Falso Negativo -> custo CUSTO_APROVAR_MAU_PAGADOR
thresholds = np.linspace(0.01, 0.99, 99)
custos = []

for t in thresholds:
    y_pred = (y_proba_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    custo_total = fp * CUSTO_NEGAR_BOM_PAGADOR + fn * CUSTO_APROVAR_MAU_PAGADOR
    custos.append({
        "threshold": t, "fp": fp, "fn": fn, "tp": tp, "tn": tn,
        "custo_total": custo_total,
    })

df_custos = pd.DataFrame(custos)
melhor = df_custos.loc[df_custos["custo_total"].idxmin()]

print("\n=== Q2 — Threshold ótimo (minimiza custo esperado) ===")
print(f"Threshold ótimo: {melhor['threshold']:.2f}")
print(f"Custo total no threshold ótimo: R$ {melhor['custo_total']:,.2f}")
print(f"  Falsos Positivos (bons negados): {int(melhor['fp'])}")
print(f"  Falsos Negativos (maus aprovados): {int(melhor['fn'])}")

# Comparação com threshold "ingênuo" de 0.5
custo_05 = df_custos.loc[(df_custos["threshold"] - 0.50).abs().idxmin()]
print(f"\nCusto no threshold padrão (0.50): R$ {custo_05['custo_total']:,.2f}")
print(f"Economia ao usar threshold ótimo: R$ {custo_05['custo_total'] - melhor['custo_total']:,.2f}")

plt.figure(figsize=(8, 5))
plt.plot(df_custos["threshold"], df_custos["custo_total"])
plt.axvline(melhor["threshold"], color="red", linestyle="--", label=f"Ótimo = {melhor['threshold']:.2f}")
plt.xlabel("Threshold de probabilidade")
plt.ylabel("Custo total esperado (R$)")
plt.title("Custo esperado vs. Threshold (custo assimétrico)")
plt.legend()
plt.tight_layout()
plt.savefig("/tmp/threshold_custo.png", dpi=150)
plt.close()

THRESHOLD_OTIMO = float(melhor["threshold"])

# COMMAND ----------
# =========================
# Q3. ESTABILIDADE / VIÉS POR FAIXA DE RENDA E IDADE
# =========================
df_val = X_test.copy()
df_val["y_true"] = y_test.values
df_val["y_proba"] = y_proba_test
df_val["y_pred"] = (df_val["y_proba"] >= THRESHOLD_OTIMO).astype(int)

# Faixas (ajuste os bins conforme a distribuição real da base)
df_val["faixa_idade"] = pd.cut(
    df_val["age"], bins=[18, 30, 40, 50, 60, 100],
    labels=["18-30", "31-40", "41-50", "51-60", "60+"]
)
df_val["faixa_renda"] = pd.qcut(
    df_val["monthly_income"].fillna(0), q=5, duplicates="drop",
    labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"]
)


def metricas_por_grupo(df_val, coluna_grupo):
    linhas = []
    for grupo, sub in df_val.groupby(coluna_grupo, observed=True):
        tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan   # bons pagadores negados indevidamente
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan   # maus pagadores aprovados indevidamente
        acc = (tp + tn) / len(sub) if len(sub) > 0 else np.nan
        linhas.append({
            "grupo": grupo, "n": len(sub), "acuracia": acc,
            "fpr_bons_negados": fpr, "fnr_maus_aprovados": fnr,
        })
    return pd.DataFrame(linhas)


tabela_idade = metricas_por_grupo(df_val, "faixa_idade")
tabela_renda = metricas_por_grupo(df_val, "faixa_renda")

print("\n=== Q3 — Performance por faixa etária ===")
print(tabela_idade.to_string(index=False))
print("\n=== Q3 — Performance por faixa de renda ===")
print(tabela_renda.to_string(index=False))

# Flag de desproporcionalidade: FNR ou FPR de um grupo > média geral + 1 desvio-padrão
for nome, tabela in [("idade", tabela_idade), ("renda", tabela_renda)]:
    for metrica in ["fpr_bons_negados", "fnr_maus_aprovados"]:
        media, desvio = tabela[metrica].mean(), tabela[metrica].std()
        outliers = tabela[tabela[metrica] > media + desvio]
        if not outliers.empty:
            print(f"\n[ALERTA] Faixas de {nome} com {metrica} desproporcional (> média + 1dp):")
            print(outliers[["grupo", metrica]].to_string(index=False))

# COMMAND ----------
# =========================
# Q4. IMPACTO FINANCEIRO ESTIMADO (R$)
# =========================
# Política atual = sem modelo (aprova todo mundo -> taxa de inadimplência = taxa observada na base)
taxa_inadimplencia_sem_modelo = (
    TAXA_INADIMPLENCIA_POLITICA_ATUAL
    if TAXA_INADIMPLENCIA_POLITICA_ATUAL is not None
    else y_test.mean()
)

n_operacoes = len(y_test)
perda_sem_modelo = n_operacoes * taxa_inadimplencia_sem_modelo * VALOR_MEDIO_OPERACAO

# Política com modelo, usando o threshold ótimo:
#   - operações negadas (y_pred == 1) não geram perda nem receita
#   - operações aprovadas (y_pred == 0) mantêm a taxa real de inadimplência OBSERVADA
#     dentro desse subgrupo (maus pagadores que passaram = falsos negativos)
aprovados_com_modelo = df_val[df_val["y_pred"] == 0]
n_aprovados = len(aprovados_com_modelo)
taxa_inadimplencia_com_modelo = aprovados_com_modelo["y_true"].mean() if n_aprovados > 0 else 0.0
perda_com_modelo = n_aprovados * taxa_inadimplencia_com_modelo * VALOR_MEDIO_OPERACAO

# Custo de oportunidade: bons pagadores negados (FP) deixam de gerar margem
bons_negados = df_val[(df_val["y_pred"] == 1) & (df_val["y_true"] == 0)]
custo_oportunidade = len(bons_negados) * CUSTO_NEGAR_BOM_PAGADOR

reducao_bruta_inadimplencia = perda_sem_modelo - perda_com_modelo
impacto_liquido = reducao_bruta_inadimplencia - custo_oportunidade

print("\n=== Q4 — Impacto financeiro estimado ===")
print(f"Operações analisadas: {n_operacoes}")
print(f"Taxa de inadimplência sem modelo (aprova todos): {taxa_inadimplencia_sem_modelo:.2%}")
print(f"Perda estimada sem modelo: R$ {perda_sem_modelo:,.2f}")
print(f"Taxa de inadimplência com modelo (apenas aprovados): {taxa_inadimplencia_com_modelo:.2%}")
print(f"Perda estimada com modelo: R$ {perda_com_modelo:,.2f}")
print(f"Redução bruta de inadimplência: R$ {reducao_bruta_inadimplencia:,.2f}")
print(f"Custo de oportunidade (bons pagadores negados): R$ {custo_oportunidade:,.2f}")
print(f"Impacto financeiro líquido estimado: R$ {impacto_liquido:,.2f}")

# COMMAND ----------
# =========================
# 5. LOG NO MLFLOW (rastreabilidade da fase de validação)
# =========================
with mlflow.start_run(run_name="validacao_shap_threshold_vies_roi"):
    mlflow.log_param("threshold_otimo", THRESHOLD_OTIMO)
    mlflow.log_param("custo_aprovar_mau_pagador", CUSTO_APROVAR_MAU_PAGADOR)
    mlflow.log_param("custo_negar_bom_pagador", CUSTO_NEGAR_BOM_PAGADOR)
    mlflow.log_metric("custo_total_threshold_otimo", float(melhor["custo_total"]))
    mlflow.log_metric("impacto_financeiro_liquido", float(impacto_liquido))
    mlflow.log_metric("taxa_inadimplencia_sem_modelo", float(taxa_inadimplencia_sem_modelo))
    mlflow.log_metric("taxa_inadimplencia_com_modelo", float(taxa_inadimplencia_com_modelo))
    mlflow.log_dict(ranking_shap.to_dict(orient="records"), "shap_ranking.json")
    mlflow.log_dict(tabela_idade.to_dict(orient="records"), "vies_por_idade.json")
    mlflow.log_dict(tabela_renda.to_dict(orient="records"), "vies_por_renda.json")
    mlflow.log_artifact("/tmp/shap_summary.png")
    mlflow.log_artifact("/tmp/shap_bar.png")
    mlflow.log_artifact("/tmp/threshold_custo.png")
    mlflow.log_param("h1_status", status_h1)
    mlflow.log_param("h2_status", status_h2)

print("\nValidação concluída e registrada no MLflow.")
