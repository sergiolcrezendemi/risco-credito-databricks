# Databricks notebook source
%python
# Databricks notebook source
# =============================================================================
# notebooks/monitoracao/02_concept_drift.py
# -----------------------------------------------------------------------------
# CONCEPT DRIFT: a relação entre as features e o target mudou? (o modelo
# passou a errar mais, mesmo que os dados de entrada pareçam parecidos —
# isso é o que 01_drift_dados.py NÃO consegue ver).
#
# Limitação real deste dataset: TARGET_COL tem horizonte de 2 anos
# (SeriousDlqin2yrs = inadimplência nos PRÓXIMOS 24 meses). Isso significa
# que o resultado real de um cliente pontuado hoje só é conhecido daqui a
# ~2 anos — não dá pra medir concept drift de verdade sem esperar o rótulo
# atrasado chegar. Duas fontes de dado, portanto:
#
#   REAL_MODE  = True   -> espera uma tabela com o resultado REALIZADO dos
#                          clientes pontuados no passado (ver seção 1B).
#                          Ainda não existe no projeto — este notebook detecta
#                          a ausência e cai para simulação automaticamente.
#   SIMULATION_MODE      -> gera um lote sintético com uma relação
#                          feature->target deliberadamente diferente da de
#                          treino, para testar se a lógica de alerta
#                          funciona ANTES de depender de rótulos reais.
#                          Lógica validada em conversas anteriores deste
#                          projeto (bugs de referência/baseline já corrigidos).
# =============================================================================

# COMMAND ----------
# =========================
# 0. CONFIGURAÇÃO
# =========================
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from mlflow.tracking import MlflowClient

logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.text("simulation_mode", "true")  # "false" quando a tabela de rótulos atrasados existir
CATALOG = dbutils.widgets.get("catalog")
SIMULATION_MODE = dbutils.widgets.get("simulation_mode").lower() == "true"

SCHEMA = "gold"
TARGET_COL = "target_dlq_2yrs"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.credito_risco_xgb"
OUTPUT_TABLE = f"{CATALOG}.ml.monitoramento_concept_drift"
MONITORING_EXPERIMENT = "/Shared/credito_risco_monitoramento"
# Tabela que ainda não existe no projeto — quando o processo de reconciliação
# de resultados reais for criado, ela deve ter (customer_id, resultado_realizado).
TABELA_RESULTADOS_REALIZADOS = f"{CATALOG}.{SCHEMA}.resultados_realizados"

FEATURE_COLS = [
    "age", "num_dependents", "monthly_income", "debt_ratio", "revolving_utilization",
    "num_open_credit_lines", "num_real_estate_loans", "num_times_30_59_days_late",
    "num_times_60_89_days_late", "num_times_90_days_late", "total_delinquency_events",
]

QUEDA_AUC_ALERTA = 0.03  # queda absoluta de AUC-ROC vs. baseline que dispara alerta
N_SIMULADO = 20_000

mlflow.set_experiment(MONITORING_EXPERIMENT)

# COMMAND ----------
# =========================
# 1. CARREGA O @champion
# =========================
client = MlflowClient()
versao_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
modelo = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
print(f"Champion carregado: {MODEL_NAME} versão {versao_champion.version}")

# COMMAND ----------
# =========================
# 1B. TENTA MODO REAL — cai para simulação se a tabela de rótulos não existir
# =========================
df_avaliacao = None
auc_referencia = None

if not SIMULATION_MODE:
    try:
        df_avaliacao = spark.table(TABELA_RESULTADOS_REALIZADOS).toPandas()
        print(f"[REAL_MODE] {len(df_avaliacao)} resultados realizados carregados de {TABELA_RESULTADOS_REALIZADOS}.")
        # AUC de referência = a do próprio treino, registrada como tag no momento do registro do modelo.
        auc_referencia = float(client.get_model_version(MODEL_NAME, versao_champion.version).tags.get("auc_roc", "nan"))
        if np.isnan(auc_referencia):
            print("[AVISO] Tag 'auc_roc' não encontrada na versão do modelo — "
                  "adicione client.set_model_version_tag(..., 'auc_roc', ...) no notebook de treino.")
    except Exception as e:
        print(f"[AVISO] {TABELA_RESULTADOS_REALIZADOS} não existe ainda ({e}). "
              f"Sem rótulos reais atrasados, não dá pra medir concept drift de verdade. "
              f"Caindo para SIMULATION_MODE para validar a lógica de alerta.")
        SIMULATION_MODE = True

# COMMAND ----------
# =========================
# 2. GERADOR SINTÉTICO (só roda em SIMULATION_MODE)
# -----------------------------------------------------------------------------
# Gera uma referência (relação feature->target igual à de treino) e um lote
# "deslocado" (relação diferente) do MESMO gerador — é essencial vir do
# mesmo gerador dos dois lados, senão o que se mede é "sintético vs. real",
# não concept drift de verdade.
# =============================================================================
def gerar_lote_sintetico(n: int, concept_drift: bool, seed: int) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    age = r.normal(45, 12, n).clip(21, 90)
    monthly_income = r.lognormal(8.6, 0.6, n)
    debt_ratio = r.gamma(2.0, 0.25, n).clip(0, 5)
    revolving_utilization = r.beta(2, 5, n).clip(0, 1.5)
    num_times_30_59 = r.poisson(0.35, n).clip(0, 10)
    num_times_60_89 = r.poisson(0.12, n).clip(0, 10)
    num_times_90 = r.poisson(0.08, n).clip(0, 10)
    num_open_credit_lines = r.poisson(8, n).clip(0, 30)

    # Relação feature->target: "concept_drift=True" enfraquece o peso de
    # debt_ratio/revolving_utilization (as duas mais preditivas no modelo real,
    # ver README Q1) — simula o cenário em que o que antes previa risco
    # deixou de prever tão bem.
    coef_debt = 2.2 if not concept_drift else 0.7
    coef_revolv = 2.6 if not concept_drift else 0.8
    logit = (
        -3.6 + coef_debt * debt_ratio + coef_revolv * revolving_utilization
        + 0.55 * num_times_30_59 + 0.85 * num_times_60_89 + 1.05 * num_times_90
        - 0.00002 * monthly_income - 0.01 * age
    )
    target = r.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "age": age, "num_dependents": r.poisson(0.9, n).clip(0, 8), "monthly_income": monthly_income,
        "debt_ratio": debt_ratio, "revolving_utilization": revolving_utilization,
        "num_open_credit_lines": num_open_credit_lines, "num_real_estate_loans": r.poisson(1.0, n).clip(0, 6),
        "num_times_30_59_days_late": num_times_30_59, "num_times_60_89_days_late": num_times_60_89,
        "num_times_90_days_late": num_times_90,
        "total_delinquency_events": num_times_30_59 + num_times_60_89 + num_times_90,
        TARGET_COL: target,
    })

if SIMULATION_MODE:
    df_referencia_sim = gerar_lote_sintetico(N_SIMULADO, concept_drift=False, seed=1)
    df_avaliacao = gerar_lote_sintetico(N_SIMULADO, concept_drift=True, seed=2)

    proba_referencia = modelo.predict_proba(df_referencia_sim[FEATURE_COLS])[:, 1]
    auc_referencia = float(roc_auc_score(df_referencia_sim[TARGET_COL], proba_referencia))

    print(f"[SIMULATION_MODE] Referência sintética (sem drift): AUC-ROC = {auc_referencia:.4f}")
    print("Essa AUC não é comparável 1:1 ao baseline real de treino — a relação "
          "feature->target sintética é mais simples. O que importa é a QUEDA "
          "relativa entre referência e lote avaliado, não o valor absoluto.")

# COMMAND ----------
# =========================
# 3. AVALIA O LOTE ATUAL CONTRA A REFERÊNCIA
# =========================
proba_atual = modelo.predict_proba(df_avaliacao[FEATURE_COLS])[:, 1]
auc_atual = float(roc_auc_score(df_avaliacao[TARGET_COL], proba_atual))
ks_atual = float(ks_2samp(
    proba_atual[df_avaliacao[TARGET_COL] == 1], proba_atual[df_avaliacao[TARGET_COL] == 0]
).statistic)

queda_auc = auc_referencia - auc_atual

print(f"\nAUC-ROC referência: {auc_referencia:.4f}")
print(f"AUC-ROC lote atual: {auc_atual:.4f}")
print(f"Queda de AUC-ROC:   {queda_auc:+.4f}")
print(f"KS-statistic lote atual: {ks_atual:.4f}")

if queda_auc >= QUEDA_AUC_ALERTA:
    print(f"\n[ALERTA] CONCEPT DRIFT — queda de AUC-ROC ({queda_auc:.4f}) >= limiar ({QUEDA_AUC_ALERTA}).")
    print("A relação entre as features e o target parece ter mudado. Considere retreino.")
else:
    print(f"\nQueda de AUC-ROC dentro do esperado (< {QUEDA_AUC_ALERTA}). Sem alerta de concept drift.")

# COMMAND ----------
# =========================
# 4. LOGA NO MLFLOW E PERSISTE HISTÓRICO
# =========================
with mlflow.start_run(run_name="concept_drift"):
    mlflow.log_param("simulation_mode", SIMULATION_MODE)
    mlflow.log_metric("auc_referencia", auc_referencia)
    mlflow.log_metric("auc_atual", auc_atual)
    mlflow.log_metric("queda_auc", queda_auc)
    mlflow.log_metric("ks_atual", ks_atual)
    mlflow.log_metric("alerta_concept_drift", int(queda_auc >= QUEDA_AUC_ALERTA))

df_resultado = pd.DataFrame([{
    "timestamp_execucao": datetime.now(timezone.utc),
    "simulation_mode": SIMULATION_MODE,
    "auc_referencia": auc_referencia,
    "auc_atual": auc_atual,
    "queda_auc": queda_auc,
    "ks_atual": ks_atual,
    "alerta": bool(queda_auc >= QUEDA_AUC_ALERTA),
}])
spark.createDataFrame(df_resultado).write.mode("append").saveAsTable(OUTPUT_TABLE)

print(f"\nResultado gravado em {OUTPUT_TABLE} e logado no experimento '{MONITORING_EXPERIMENT}'.")
