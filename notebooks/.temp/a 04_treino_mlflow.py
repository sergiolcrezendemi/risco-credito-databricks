# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Treino: XGBoost vs. Regressão Logística (MLflow + Unity Catalog Model Registry)
# MAGIC
# MAGIC **Case:** `risco-credito-databricks`
# MAGIC
# MAGIC Cobre o item 4 do roadmap do README: MLflow tracking de experimentos, comparação XGBoost vs.
# MAGIC Regressão Logística, explicabilidade via SHAP, e a decisão de qual modelo vai para produção
# MAGIC **documentada por métrica, não assumida de antemão**.
# MAGIC
# MAGIC **Tratamento do desbalanceamento (~6,7% positivo) — decisão documentada:** usei *class
# MAGIC weighting* nativo de cada modelo (`class_weight="balanced"` na Regressão Logística,
# MAGIC `scale_pos_weight` no XGBoost) em vez de SMOTE. Motivo: SMOTE gera exemplos sintéticos
# MAGIC interpolando entre vizinhos da classe minoritária no espaço de features — em dado financeiro
# MAGIC real, isso pode criar combinações de `DebtRatio`/`RevolvingUtilizationOfUnsecuredLines`/etc.
# MAGIC que não correspondem a nenhum perfil real de cliente, distorcendo o que o modelo aprende sobre
# MAGIC a fronteira de decisão. *Class weighting* não inventa dado — só reponderā o custo do erro na
# MAGIC função de perda, mantendo o modelo honesto sobre a distribuição real observada.
# MAGIC
# MAGIC **Promoção champion/challenger:** o modelo novo entra sempre como `@challenger`. Só vira
# MAGIC `@champion` se superar o campeão atual em AUC-ROC no mesmo conjunto de teste — sem isso, um
# MAGIC retraining ruim nunca substitui o que já está funcionando em produção.

# COMMAND ----------

dbutils.widgets.text("catalog", "db_risco-credito-databricks", "Catálogo")
catalog = dbutils.widgets.get("catalog")
print(f"catalog = {catalog}")

# COMMAND ----------

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from mlflow import MlflowClient
from mlflow.models import infer_signature
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.models.risco_credito_score"
experiment_name = "/Shared/risco-credito-databricks/treino"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregando treino/teste (Gold)
# MAGIC
# MAGIC `.toPandas()` é aceitável aqui — ~150 mil linhas cabe folgado em memória do driver. Se o volume
# MAGIC crescer ordens de grandeza, isso vira o primeiro ponto a revisar (treino distribuído via
# MAGIC `xgboost.spark` em vez de scikit-learn puro).

# COMMAND ----------

df_train = spark.table(f"`{catalog}`.`gold`.`credito_train`").toPandas()
df_test = spark.table(f"`{catalog}`.`gold`.`credito_test`").toPandas()

feature_cols = [
    "revolving_utilization", "age", "times_30_59_days_late", "debt_ratio",
    "monthly_income", "open_credit_lines", "times_90_days_late", "real_estate_lines",
    "times_60_89_days_late", "dependents", "is_outlier_utilization", "monthly_income_imputed",
    "dependents_imputed", "total_times_late", "has_dependents", "income_per_dependent",
    "log_monthly_income", "credit_lines_per_decade_of_age",
]

X_train, y_train = df_train[feature_cols], df_train["target"]
X_test, y_test = df_test[feature_cols], df_test["target"]

taxa_positiva = y_train.mean()
scale_pos_weight = (1 - taxa_positiva) / taxa_positiva

print(f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas")
print(f"Taxa positiva (treino): {taxa_positiva:.4f} | scale_pos_weight: {scale_pos_weight:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Métricas de avaliação — funções reaproveitadas nos dois modelos

# COMMAND ----------

def calcular_ks(y_true, y_proba):
    """KS-statistic: maior distância entre as distribuições cumulativas de score dos bons e maus pagadores."""
    score_positivos = y_proba[y_true == 1]
    score_negativos = y_proba[y_true == 0]
    return ks_2samp(score_positivos, score_negativos).statistic

def recall_no_precision_alvo(y_true, y_proba, precisao_alvo=0.30):
    """
    Recall no maior threshold que ainda entrega a precisão mínima definida pelo negócio.
    precisao_alvo=0.30 é um ponto de partida documentado — revisar com a área de negócio antes
    de qualquer uso além desta POC (o custo real de falso positivo/negativo não está modelado aqui).
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    indices_validos = np.where(precisions[:-1] >= precisao_alvo)[0]
    if len(indices_validos) == 0:
        return 0.0
    return recalls[indices_validos].max()

def avaliar(y_true, y_proba):
    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "ks_statistic": calcular_ks(y_true, y_proba),
        "recall_at_precision_30": recall_no_precision_alvo(y_true, y_proba, 0.30),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo 1 — Regressão Logística (baseline interpretável)

# COMMAND ----------

with mlflow.start_run(run_name="logistic_regression_baseline") as run_lr:
    pipeline_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline_lr.fit(X_train, y_train)

    proba_test_lr = pipeline_lr.predict_proba(X_test)[:, 1]
    metricas_lr = avaliar(y_test, proba_test_lr)

    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metrics(metricas_lr)

    signature = infer_signature(X_train, pipeline_lr.predict_proba(X_train))
    mlflow.sklearn.log_model(
        pipeline_lr, "model", signature=signature, input_example=X_train.head(5)
    )

    run_id_lr = run_lr.info.run_id

print("Regressão Logística —", metricas_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelo 2 — XGBoost (principal)

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_principal") as run_xgb:
    params_xgb = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
    }
    modelo_xgb = xgb.XGBClassifier(**params_xgb)
    modelo_xgb.fit(X_train, y_train)

    proba_test_xgb = modelo_xgb.predict_proba(X_test)[:, 1]
    metricas_xgb = avaliar(y_test, proba_test_xgb)

    mlflow.log_params(params_xgb)
    mlflow.log_metrics(metricas_xgb)

    signature = infer_signature(X_train, modelo_xgb.predict_proba(X_train))
    mlflow.xgboost.log_model(
        modelo_xgb, "model", signature=signature, input_example=X_train.head(5)
    )

    # SHAP — obrigatório para o modelo principal (README). TreeExplainer é exato e rápido para
    # modelos baseados em árvore, ao contrário do KernelExplainer (aproximado, genérico).
    explainer = shap.TreeExplainer(modelo_xgb)
    shap_values = explainer.shap_values(X_test.sample(min(2000, len(X_test)), random_state=42))

    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X_test.sample(min(2000, len(X_test)), random_state=42), show=False)
    mlflow.log_figure(plt.gcf(), "shap_summary.png")
    plt.close()

    run_id_xgb = run_xgb.info.run_id

print("XGBoost —", metricas_xgb)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparação e checagem das hipóteses (H1/H2) via SHAP
# MAGIC
# MAGIC A importância média absoluta de SHAP no XGBoost responde diretamente a pergunta de
# MAGIC investigação 1 do README — qual variável pesa mais na decisão do modelo.

# COMMAND ----------

importancia_shap = pd.DataFrame({
    "feature": feature_cols,
    "importancia_media_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("importancia_media_abs_shap", ascending=False)

display(importancia_shap)

print("\n--- Comparação de métricas (conjunto de teste) ---")
comparacao = pd.DataFrame([
    {"modelo": "Regressão Logística", **metricas_lr},
    {"modelo": "XGBoost", **metricas_xgb},
])
display(comparacao)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decisão de qual modelo vai para produção
# MAGIC
# MAGIC Critério simples e documentado: maior AUC-ROC no teste vence. Empate (diferença < 0.005) cairia
# MAGIC para a Regressão Logística por interpretabilidade — não é o caso normalmente esperado aqui, mas
# MAGIC a regra de desempate fica registrada de qualquer forma.

# COMMAND ----------

MARGEM_EMPATE = 0.005

if metricas_xgb["auc_roc"] - metricas_lr["auc_roc"] > MARGEM_EMPATE:
    modelo_vencedor, run_id_vencedor, metricas_vencedoras = "xgboost", run_id_xgb, metricas_xgb
elif metricas_lr["auc_roc"] - metricas_xgb["auc_roc"] > MARGEM_EMPATE:
    modelo_vencedor, run_id_vencedor, metricas_vencedoras = "logistic_regression", run_id_lr, metricas_lr
else:
    modelo_vencedor, run_id_vencedor, metricas_vencedoras = "logistic_regression", run_id_lr, metricas_lr
    print("Empate técnico (diferença < 0.005 AUC) — desempate por interpretabilidade: Regressão Logística.")

print(f"Modelo vencedor: {modelo_vencedor} | AUC-ROC: {metricas_vencedoras['auc_roc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registro no Unity Catalog Model Registry — challenger, com promoção condicional a champion

# COMMAND ----------

client = MlflowClient()

model_uri = f"runs:/{run_id_vencedor}/model"
versao_registrada = mlflow.register_model(model_uri, model_name)

client.set_registered_model_alias(model_name, "challenger", versao_registrada.version)
for chave, valor in metricas_vencedoras.items():
    client.set_model_version_tag(model_name, versao_registrada.version, chave, str(valor))
client.set_model_version_tag(model_name, versao_registrada.version, "algoritmo", modelo_vencedor)

print(f"Versão {versao_registrada.version} registrada como @challenger em {model_name}")

# COMMAND ----------

try:
    versao_champion_atual = client.get_model_version_by_alias(model_name, "champion")
    auc_champion_atual = float(
        client.get_model_version(model_name, versao_champion_atual.version).tags.get("auc_roc", "0")
    )
    print(f"Champion atual: versão {versao_champion_atual.version}, AUC-ROC {auc_champion_atual:.4f}")

    if metricas_vencedoras["auc_roc"] > auc_champion_atual:
        client.set_registered_model_alias(model_name, "champion", versao_registrada.version)
        print(f"Challenger superou o champion — versão {versao_registrada.version} promovida a @champion.")
    else:
        print("Challenger não superou o champion atual — @champion mantido, @challenger só fica registrado para inspeção.")

except mlflow.exceptions.RestException:
    # Nenhum @champion ainda existe (primeira execução) — o challenger vira champion direto
    client.set_registered_model_alias(model_name, "champion", versao_registrada.version)
    print(f"Nenhum @champion prévio — versão {versao_registrada.version} promovida a @champion diretamente.")
