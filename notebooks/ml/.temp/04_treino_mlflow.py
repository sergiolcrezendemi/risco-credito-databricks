# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# =============================================================================
# 04_treino_mlflow.py
# -----------------------------------------------------------------------------
# Fase de Treino do case "Risco de Crédito" (item 4 do roadmap do README.md)
#
# O que este notebook faz:
#   1. Carrega a base Gold (JOIN fct_credit_profile + dim_customer)
#   2. Treina e compara DOIS modelos: Regressão Logística (baseline interpretável)
#      e XGBoost (modelo principal) — desbalanceamento tratado via *class weighting*
#      nativo de cada modelo, não SMOTE (decisão documentada abaixo)
#   3. Avalia os dois no mesmo conjunto de teste: AUC-ROC, KS-statistic, recall no
#      threshold que entrega uma precisão-alvo definida pelo negócio
#   4. Gera SHAP (TreeExplainer) para o XGBoost — a tabela de importância resultante
#      é a mesma pergunta de investigação respondida no notebook 03 (H1/H2)
#   5. Decide o modelo vencedor por REGRA (maior AUC-ROC, com margem de empate
#      técnico caindo para o modelo interpretável) — não por preferência
#   6. Registra o vencedor no Unity Catalog Model Registry como @challenger;
#      promove a @champion só se superar o champion atual no mesmo teste
#
# Por que fica fora da pipeline declarativa (Lakeflow):
#   Treino tem efeito colateral (chamadas ao MLflow Client, registro de modelo) —
#   não é uma transformação pura de DataFrame, então não se encaixa no modelo
#   declarativo usado nos notebooks 02/03 de Silver/Gold.
#
# Decisão de desbalanceamento (documentada, não assumida):
#   A base tem ~6,7% de positivos (inadimplentes). Usamos class_weight="balanced"
#   na Regressão Logística e scale_pos_weight no XGBoost, em vez de SMOTE — SMOTE
#   poderia gerar combinações sintéticas de debt_ratio/revolving_utilization sem
#   correspondência real em um cliente de crédito.
# =============================================================================

# COMMAND ----------

# =========================
# 0. CONFIGURAÇÃO
# =========================
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import logging
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.set_registry_uri("databricks-uc")

# --- Ajuste estes parâmetros ao seu ambiente ---
CATALOG = "credito_prd"
SCHEMA = "gold"
FCT_TABLE = "fct_credit_profile"
DIM_CUSTOMER_TABLE = "dim_customer"
TARGET_COL = "target_dlq_2yrs"                 # 1 = inadimplente, 0 = bom pagador
MODEL_NAME = f"{CATALOG}.{SCHEMA}.credito_risco_score"   # nome único no Registry (agnóstico ao algoritmo)

# --- Precisão-alvo para a métrica de recall (Q2 do README) ---
# Ponto de partida documentado — validar com a área de negócio antes de qualquer uso
# além desta POC (o custo real de falso positivo/negativo é tratado no notebook 03).
PRECISAO_ALVO = 0.30

# --- Regra de decisão entre modelos ---
MARGEM_EMPATE_TECNICO = 0.005   # se a diferença de AUC-ROC for menor que isto, cai para o interpretável

# COMMAND ----------

# =========================
# 1. CARGA DOS DADOS (mesma base do notebook 03)
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

n_antes = len(df)
df = df.dropna(subset=[TARGET_COL])
n_removidos = n_antes - len(df)
if n_removidos > 0:
    print(f"[AVISO] {n_removidos} de {n_antes} registros removidos por '{TARGET_COL}' nulo "
          f"({n_removidos / n_antes:.2%}).")
df[TARGET_COL] = df[TARGET_COL].astype(int)

FEATURE_COLS = [
    "age", "num_dependents", "monthly_income", "debt_ratio", "revolving_utilization",
    "num_open_credit_lines", "num_real_estate_loans", "num_times_30_59_days_late",
    "num_times_60_89_days_late", "num_times_90_days_late", "total_delinquency_events",
]

X = df[FEATURE_COLS]
y = df[TARGET_COL]

nulos_por_coluna = X.isna().sum()
nulos_por_coluna = nulos_por_coluna[nulos_por_coluna > 0]
if not nulos_por_coluna.empty:
    print("[AVISO] Colunas com valores nulos nas features (imputadas com mediana para a "
          "Regressão Logística; o XGBoost lida com NaN nativamente):")
    for col, qtd in nulos_por_coluna.items():
        print(f"  {col}: {qtd} ({qtd / len(X):.2%})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

taxa_positiva = y_train.mean()
scale_pos_weight = (1 - taxa_positiva) / taxa_positiva

print(f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas")
print(f"Taxa positiva (treino): {taxa_positiva:.4f} | scale_pos_weight: {scale_pos_weight:.2f}")

# COMMAND ----------

# =========================
# 2. MÉTRICAS DE AVALIAÇÃO — funções reaproveitadas nos dois modelos
# =========================
def calcular_ks(y_true, y_proba):
    """KS-statistic: maior distância entre as distribuições cumulativas de score
    dos bons e maus pagadores. Complementa o AUC-ROC mostrando o ponto de melhor
    separação entre as duas classes."""
    score_positivos = y_proba[y_true == 1]
    score_negativos = y_proba[y_true == 0]
    return ks_2samp(score_positivos, score_negativos).statistic


def recall_no_precision_alvo(y_true, y_proba, precisao_alvo=PRECISAO_ALVO):
    """Recall no maior threshold que ainda entrega a precisão mínima definida pelo
    negócio. Se nenhum threshold atinge a precisão-alvo, retorna 0.0 e sinaliza."""
    precisao, recall, _ = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve retorna precisao e recall com o MESMO tamanho
    # (ambos len(thresholds)+1) — sem fatiar, ou o índice booleano desalinha.
    candidatos = recall[precisao >= precisao_alvo]
    if len(candidatos) == 0:
        return 0.0
    return float(candidatos.max())


def avaliar(y_true, y_proba):
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "ks_statistic": float(calcular_ks(y_true.values, y_proba)),
        f"recall_em_{int(PRECISAO_ALVO*100)}pct_precisao": float(
            recall_no_precision_alvo(y_true.values, y_proba)
        ),
    }

# COMMAND ----------

# =========================
# 3. MODELO 1 — REGRESSÃO LOGÍSTICA (baseline interpretável)
# =========================
with mlflow.start_run(run_name="logistic_regression") as run_lr:
    pipeline_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # LogisticRegression não aceita NaN nativamente
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])
    pipeline_lr.fit(X_train, y_train)
    proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
    metricas_lr = avaliar(y_test, proba_lr)

    mlflow.log_param("algoritmo", "logistic_regression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("imputer_strategy", "median")
    for chave, valor in metricas_lr.items():
        mlflow.log_metric(chave, valor)

    assinatura_lr = infer_signature(X_train, pipeline_lr.predict_proba(X_train)[:, 1])
    mlflow.sklearn.log_model(
        pipeline_lr, "model",
        signature=assinatura_lr, input_example=X_train.head(5),
    )
    run_id_lr = run_lr.info.run_id

print(f"[Regressão Logística] AUC-ROC: {metricas_lr['auc_roc']:.4f} | "
      f"KS: {metricas_lr['ks_statistic']:.4f} | "
      f"Recall@{int(PRECISAO_ALVO*100)}%precisão: "
      f"{metricas_lr[f'recall_em_{int(PRECISAO_ALVO*100)}pct_precisao']:.4f}")

# COMMAND ----------

# =========================
# 4. MODELO 2 — XGBOOST (modelo principal)
# =========================
with mlflow.start_run(run_name="xgboost") as run_xgb:
    modelo_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    modelo_xgb.fit(X_train, y_train)
    proba_xgb = modelo_xgb.predict_proba(X_test)[:, 1]
    metricas_xgb = avaliar(y_test, proba_xgb)

    mlflow.log_param("algoritmo", "xgboost")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    for chave, valor in metricas_xgb.items():
        mlflow.log_metric(chave, valor)

    # SHAP — responde à mesma pergunta de investigação validada no notebook 03 (H1/H2)
    explainer = shap.TreeExplainer(modelo_xgb)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("/tmp/shap_summary_treino.png", dpi=150)
    plt.close()
    mlflow.log_artifact("/tmp/shap_summary_treino.png")

    assinatura_xgb = infer_signature(X_train, modelo_xgb.predict_proba(X_train)[:, 1])
    mlflow.xgboost.log_model(
        modelo_xgb, "model",
        signature=assinatura_xgb, input_example=X_train.head(5),
    )
    run_id_xgb = run_xgb.info.run_id

print(f"[XGBoost] AUC-ROC: {metricas_xgb['auc_roc']:.4f} | "
      f"KS: {metricas_xgb['ks_statistic']:.4f} | "
      f"Recall@{int(PRECISAO_ALVO*100)}%precisão: "
      f"{metricas_xgb[f'recall_em_{int(PRECISAO_ALVO*100)}pct_precisao']:.4f}")

# COMMAND ----------

# =========================
# 5. DECISÃO DO MODELO VENCEDOR (por regra, não por preferência)
# =========================
diferenca_auc = metricas_xgb["auc_roc"] - metricas_lr["auc_roc"]

if abs(diferenca_auc) < MARGEM_EMPATE_TECNICO:
    # Empate técnico: preferimos o modelo interpretável (Regressão Logística)
    modelo_vencedor = "logistic_regression"
    run_id_vencedor = run_id_lr
    metricas_vencedoras = metricas_lr
    flavor_vencedor = mlflow.sklearn
else:
    modelo_vencedor = "xgboost" if diferenca_auc > 0 else "logistic_regression"
    run_id_vencedor = run_id_xgb if modelo_vencedor == "xgboost" else run_id_lr
    metricas_vencedoras = metricas_xgb if modelo_vencedor == "xgboost" else metricas_lr
    flavor_vencedor = mlflow.xgboost if modelo_vencedor == "xgboost" else mlflow.sklearn

print(f"\nDiferença de AUC-ROC (XGBoost - Regressão Logística): {diferenca_auc:+.4f} "
      f"(margem de empate técnico: {MARGEM_EMPATE_TECNICO})")
print(f"Modelo vencedor: {modelo_vencedor} | AUC-ROC: {metricas_vencedoras['auc_roc']:.4f}")

# COMMAND ----------

# =========================
# 6. REGISTRO NO UNITY CATALOG MODEL REGISTRY — challenger, com promoção condicional a champion
# =========================
client = MlflowClient()

model_uri = f"runs:/{run_id_vencedor}/model"
versao_registrada = mlflow.register_model(model_uri, MODEL_NAME)

client.set_registered_model_alias(MODEL_NAME, "challenger", versao_registrada.version)
for chave, valor in metricas_vencedoras.items():
    client.set_model_version_tag(MODEL_NAME, versao_registrada.version, chave, str(valor))
client.set_model_version_tag(MODEL_NAME, versao_registrada.version, "algoritmo", modelo_vencedor)

print(f"Versão {versao_registrada.version} registrada como @challenger em {MODEL_NAME}")

# COMMAND ----------

try:
    versao_champion_atual = client.get_model_version_by_alias(MODEL_NAME, "champion")
    auc_champion_atual = float(
        client.get_model_version(MODEL_NAME, versao_champion_atual.version).tags.get("auc_roc", "0")
    )
    print(f"Champion atual: versão {versao_champion_atual.version}, AUC-ROC {auc_champion_atual:.4f}")

    if metricas_vencedoras["auc_roc"] > auc_champion_atual:
        client.set_registered_model_alias(MODEL_NAME, "champion", versao_registrada.version)
        print(f"Challenger superou o champion — versão {versao_registrada.version} promovida a @champion.")
    else:
        print("Challenger não superou o champion atual — @champion mantido, "
              "@challenger só fica registrado para inspeção.")

except mlflow.exceptions.RestException:
    # Nenhum @champion ainda existe (primeira execução) — o challenger vira champion direto
    client.set_registered_model_alias(MODEL_NAME, "champion", versao_registrada.version)
    print(f"Nenhum @champion prévio — versão {versao_registrada.version} promovida a @champion diretamente.")

print("\nTreino concluído e registrado no MLflow / Unity Catalog Model Registry.")