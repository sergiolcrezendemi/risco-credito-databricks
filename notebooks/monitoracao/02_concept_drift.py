# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# =============================================================================
# notebooks/monitoracao/02_concept_drift.py
# -----------------------------------------------------------------------------
# CONCEPT DRIFT: a relação entre as features e o target mudou? (o modelo
# passou a errar mais, mesmo que os dados de entrada pareçam parecidos —
# isso é o que 01_drift_dados.py NÃO consegue ver).
#
# Limitação real deste dataset: TARGET_COL tem horizonte de 2 anos
# (SeriousDlqin2yrs = inadimplência nos PRÓXIMOS 24 meses) e a base de scoring
# (cs-test.csv) NÃO tem rótulo. Ou seja: AUC só pode ser medida em base com
# target. Por isso existem três modos de avaliação (widget `modo`):
#
#   simulacao              -> lote sintético com relação feature->target
#                             deliberadamente diferente da de treino. Serve
#                             para provar que o ALERTA dispara (queda esperada).
#   treino_amostra         -> base de TREINO real, dividida por hash de
#                             customer_id: 80% = referência, 20% = lote atual.
#                             Serve para provar que o alerta NÃO dispara quando
#                             não há drift (queda esperada ~0). Atenção: o
#                             champion foi treinado nesses dados, então a AUC
#                             absoluta é otimista — só a QUEDA entre as duas
#                             partes é interpretável.
#   resultados_realizados  -> tabela de resultados reais atrasados (ainda não
#                             existe no projeto). Se faltar, cai para simulacao.
#
# Além disso, a seção 3B calcula DRIFT DE PREDIÇÃO (treino x scoring): compara
# a distribuição do score do champion nas duas bases via PSI. Não precisa de
# rótulo, então é a única leitura de "modelo" possível sobre a base de scoring.
# =============================================================================

# COMMAND ----------

# =========================
# 0. CONFIGURAÇÃO
# =========================
import logging
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

logging.getLogger("mlflow").setLevel(logging.ERROR)
mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "credito_dev")
dbutils.widgets.dropdown(
    "modo", "simulacao", ["simulacao", "treino_amostra", "resultados_realizados"]
)
CATALOG = dbutils.widgets.get("catalog")
MODO = dbutils.widgets.get("modo")

SCHEMA = "gold"
FCT_TABLE = "fct_credit_profile"
DIM_CUSTOMER_TABLE = "dim_customer"
TARGET_COL = "target_dlq_2yrs"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.credito_risco_xgb"
OUTPUT_TABLE = f"{CATALOG}.ml.monitoramento_concept_drift"
MONITORING_EXPERIMENT = "/Shared/credito_risco_monitoramento"
# Quando o processo de reconciliação de resultados reais existir, a tabela deve
# ter customer_id, TARGET_COL (resultado realizado) e todas as FEATURE_COLS.
TABELA_RESULTADOS_REALIZADOS = f"{CATALOG}.{SCHEMA}.resultados_realizados"

FEATURE_COLS = [
    "age",
    "num_dependents",
    "monthly_income",
    "debt_ratio",
    "revolving_utilization",
    "num_open_credit_lines",
    "num_real_estate_loans",
    "num_times_30_59_days_late",
    "num_times_60_89_days_late",
    "num_times_90_days_late",
    "total_delinquency_events",
]

QUEDA_AUC_ALERTA = 0.03  # queda absoluta de AUC-ROC vs. baseline que dispara alerta
N_SIMULADO = 20_000
PCT_AMOSTRA = 20  # % do treino usado como "lote atual" em modo treino_amostra
PSI_MODERADO = 0.10
PSI_SEVERO = 0.25

mlflow.set_experiment(MONITORING_EXPERIMENT)

# COMMAND ----------

# =========================
# 0b. IMPORT DA LÓGICA TESTÁVEL (src/)
#     coberta por tests/test_concept_drift.py e tests/test_data_drift.py
# =========================
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

from src.monitoring.concept_drift import avaliar_queda_auc, gerar_lote_sintetico
from src.monitoring.data_drift import calcular_psi, classificar_psi

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
# 1B. FUNÇÃO DE CARGA DO GOLD (usada pelo modo treino_amostra e pela seção 3B)
# =========================
# pmod garante bucket 0..99 mesmo quando o hash é negativo.
BUCKET = "pmod(xxhash64(f.customer_id), 100)"


def carregar_gold(filtro_sql: str) -> pd.DataFrame:
    df_spark = spark.sql(f"""
        SELECT c.age, c.num_dependents, f.monthly_income, f.debt_ratio, f.revolving_utilization,
               f.num_open_credit_lines, f.num_real_estate_loans, f.num_times_30_59_days_late,
               f.num_times_60_89_days_late, f.num_times_90_days_late, f.total_delinquency_events,
               f.{TARGET_COL} AS {TARGET_COL}
        FROM {CATALOG}.{SCHEMA}.{FCT_TABLE} f
        JOIN {CATALOG}.{SCHEMA}.{DIM_CUSTOMER_TABLE} c ON f.customer_id = c.customer_id
        WHERE {filtro_sql}
    """)
    return df_spark.toPandas()


# COMMAND ----------

# =========================
# 2. REFERÊNCIA E LOTE — depende do modo
# =========================
modo_efetivo = MODO
df_referencia = None
df_avaliacao = None
auc_referencia = None

if MODO == "resultados_realizados":
    try:
        df_avaliacao = spark.table(TABELA_RESULTADOS_REALIZADOS).toPandas()
        print(
            f"[resultados_realizados] {len(df_avaliacao)} resultados carregados de "
            f"{TABELA_RESULTADOS_REALIZADOS}."
        )
        # AUC de referência = a do treino, registrada como tag no registro do modelo.
        auc_referencia = float(
            client.get_model_version(MODEL_NAME, versao_champion.version).tags.get("auc_roc", "nan")
        )
        if np.isnan(auc_referencia):
            print(
                "[AVISO] Tag 'auc_roc' não encontrada na versão do modelo — "
                "adicione client.set_model_version_tag(..., 'auc_roc', ...) no notebook de treino."
            )
    except Exception as e:
        print(
            f"[AVISO] {TABELA_RESULTADOS_REALIZADOS} não existe ainda ({e}). "
            "Sem rótulos reais atrasados, não dá pra medir concept drift de verdade. "
            "Caindo para o modo 'simulacao' para validar a lógica de alerta."
        )
        modo_efetivo = "simulacao"

if modo_efetivo == "simulacao":
    df_referencia = gerar_lote_sintetico(N_SIMULADO, concept_drift=False, seed=1)
    df_avaliacao = gerar_lote_sintetico(N_SIMULADO, concept_drift=True, seed=2)
    print(
        "[simulacao] Referência sintética SEM drift x lote sintético COM drift. "
        "Resultado esperado: ALERTA."
    )
    print(
        "A AUC sintética não é comparável 1:1 ao baseline real de treino — o que "
        "importa é a QUEDA relativa entre referência e lote avaliado."
    )

elif modo_efetivo == "treino_amostra":
    filtro_treino = f"f.{TARGET_COL} IS NOT NULL"
    df_referencia = carregar_gold(f"{filtro_treino} AND {BUCKET} >= {PCT_AMOSTRA}")
    df_avaliacao = carregar_gold(f"{filtro_treino} AND {BUCKET} < {PCT_AMOSTRA}")
    if df_referencia.empty or df_avaliacao.empty:
        raise RuntimeError(
            f"[treino_amostra] Referência ({len(df_referencia)}) ou lote atual "
            f"({len(df_avaliacao)}) vazios — a Gold tem linhas de treino? "
            "Rode o job_ingestao (dataset=training) antes."
        )
    print(
        f"[treino_amostra] Referência = {len(df_referencia)} linhas ({100 - PCT_AMOSTRA}% do treino) | "
        f"Lote atual = {len(df_avaliacao)} linhas ({PCT_AMOSTRA}%). Resultado esperado: SEM alerta."
    )
    print(
        "O champion foi treinado nessas linhas, então a AUC absoluta é otimista. "
        "Só a queda entre as duas partes é interpretável."
    )

# AUC de referência: calculada sobre df_referencia, exceto no modo real
# (onde vem da tag do modelo).
if df_referencia is not None:
    proba_referencia = modelo.predict_proba(df_referencia[FEATURE_COLS])[:, 1]
    auc_referencia = float(roc_auc_score(df_referencia[TARGET_COL], proba_referencia))
    print(f"AUC-ROC da referência ({modo_efetivo}): {auc_referencia:.4f}")

# COMMAND ----------

# =========================
# 3. AVALIA O LOTE ATUAL CONTRA A REFERÊNCIA
# =========================
proba_atual = modelo.predict_proba(df_avaliacao[FEATURE_COLS])[:, 1]
auc_atual = float(roc_auc_score(df_avaliacao[TARGET_COL], proba_atual))
ks_atual = float(
    ks_2samp(
        proba_atual[df_avaliacao[TARGET_COL] == 1], proba_atual[df_avaliacao[TARGET_COL] == 0]
    ).statistic
)

resultado_avaliacao = avaliar_queda_auc(auc_referencia, auc_atual, limiar=QUEDA_AUC_ALERTA)
queda_auc = resultado_avaliacao["queda_auc"]
alerta = resultado_avaliacao["alerta"]

print(f"\nAUC-ROC referência: {auc_referencia:.4f}")
print(f"AUC-ROC lote atual: {auc_atual:.4f}")
print(f"Queda de AUC-ROC:   {queda_auc:+.4f}")
print(f"KS-statistic lote atual: {ks_atual:.4f}")

if alerta:
    print(
        f"\n[ALERTA] CONCEPT DRIFT — queda de AUC-ROC ({queda_auc:.4f}) >= limiar ({QUEDA_AUC_ALERTA})."
    )
    print("A relação entre as features e o target parece ter mudado. Considere retreino.")
else:
    print(
        f"\nQueda de AUC-ROC dentro do esperado (< {QUEDA_AUC_ALERTA}). Sem alerta de concept drift."
    )

# Leitura esperada por modo — ajuda a validar o próprio notebook
if modo_efetivo == "simulacao":
    print(f"[TESTE] Modo simulacao deve ALERTAR: {'OK' if alerta else 'FALHOU — não alertou'}")
elif modo_efetivo == "treino_amostra":
    print(f"[TESTE] Modo treino_amostra NÃO deve alertar: {'OK' if not alerta else 'FALHOU — alertou'}")

# COMMAND ----------

# =========================
# 3B. DRIFT DE PREDIÇÃO (treino x scoring) — não precisa de rótulo
# =========================
# Compara a distribuição do score do champion na base de treino com a do
# scoring. É a leitura de "modelo" possível sobre a base de scoring, que não
# tem target. Não substitui concept drift: mostra que o SCORE mudou, não que o
# modelo passou a errar.
psi_predicao = float("nan")
classificacao_predicao = "indefinido"
n_scoring = 0

df_treino_full = carregar_gold(f"f.{TARGET_COL} IS NOT NULL")
df_scoring_full = carregar_gold(f"f.{TARGET_COL} IS NULL")
n_scoring = len(df_scoring_full)

if df_treino_full.empty or df_scoring_full.empty:
    print(
        f"\n[3B] Pulado — treino ({len(df_treino_full)}) ou scoring ({n_scoring}) sem linhas no Gold. "
        "Rode o job_ingestao_scoring para popular o scoring."
    )
else:
    proba_treino = modelo.predict_proba(df_treino_full[FEATURE_COLS])[:, 1]
    proba_scoring = modelo.predict_proba(df_scoring_full[FEATURE_COLS])[:, 1]
    psi_predicao = calcular_psi(proba_treino, proba_scoring)
    classificacao_predicao = classificar_psi(psi_predicao, moderado=PSI_MODERADO, severo=PSI_SEVERO)
    print(
        f"\n[3B] Drift de predição — score médio treino = {proba_treino.mean():.4f} | "
        f"scoring = {proba_scoring.mean():.4f}"
    )
    print(f"[3B] PSI do score = {psi_predicao:.4f} ({classificacao_predicao})")

# COMMAND ----------

# =========================
# 4. LOGA NO MLFLOW E PERSISTE HISTÓRICO
# =========================
with mlflow.start_run(run_name=f"concept_drift_{modo_efetivo}"):
    mlflow.log_param("modo", MODO)
    mlflow.log_param("modo_efetivo", modo_efetivo)
    mlflow.log_param("simulation_mode", modo_efetivo == "simulacao")
    mlflow.log_metric("auc_referencia", auc_referencia)
    mlflow.log_metric("auc_atual", auc_atual)
    mlflow.log_metric("queda_auc", queda_auc)
    mlflow.log_metric("ks_atual", ks_atual)
    mlflow.log_metric("alerta_concept_drift", int(alerta))
    if not np.isnan(psi_predicao):
        mlflow.log_metric("psi_predicao_treino_vs_scoring", psi_predicao)

df_resultado = pd.DataFrame(
    [
        {
            "timestamp_execucao": datetime.now(timezone.utc),
            "simulation_mode": modo_efetivo == "simulacao",
            "modo": modo_efetivo,
            "auc_referencia": auc_referencia,
            "auc_atual": auc_atual,
            "queda_auc": queda_auc,
            "ks_atual": ks_atual,
            "alerta": bool(alerta),
            "psi_predicao": psi_predicao,
            "classificacao_psi_predicao": classificacao_predicao,
            "n_scoring": n_scoring,
        }
    ]
)

def garantir_colunas(tabela: str, colunas: dict) -> None:
    """Se a tabela já existe, adiciona as colunas que faltam (ALTER TABLE).
    Se ainda não existe, o saveAsTable abaixo a cria já com todas."""
    if not spark.catalog.tableExists(tabela):
        return
    existentes = {c.lower() for c in spark.table(tabela).columns}
    faltantes = {n: t for n, t in colunas.items() if n.lower() not in existentes}
    if faltantes:
        defs = ", ".join(f"`{n}` {t}" for n, t in faltantes.items())
        spark.sql(f"ALTER TABLE {tabela} ADD COLUMNS ({defs})")
        print(f"[schema] Colunas adicionadas em {tabela}: {list(faltantes)}")


garantir_colunas(
    OUTPUT_TABLE,
    {
        "modo": "STRING",
        "psi_predicao": "DOUBLE",
        "classificacao_psi_predicao": "STRING",
        "n_scoring": "BIGINT",
    },
)

# mergeSchema fica como segunda garantia caso surja outra coluna no futuro.
(
    spark.createDataFrame(df_resultado)
    .write.mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"\nResultado gravado em {OUTPUT_TABLE} e logado no experimento '{MONITORING_EXPERIMENT}'.")
