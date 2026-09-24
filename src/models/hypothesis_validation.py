# src/models/hypothesis_validation.py
# ==============================================================================
# VALIDAÇÃO DAS HIPÓTESES H1-H4 (ver README.md)
#
# Lê da GOLD (fct_credit_profile + dim_customer). Os valores financeiros (H4)
# vêm de src/config/business_params.py, mesma fonte de Q2/Q4.
#
# UM MODELO SÓ: as hipóteses são validadas no @champion registrado no Unity
# Catalog — o mesmo modelo usado na inferência batch, no monitoramento e na
# análise de viés. A versão anterior treinava um XGBoost próprio a cada
# execução (split de 20%, com imputação), então os números do README vinham
# de um modelo diferente do que ia para produção.
#
# UM SPLIT SÓ: `prepare_train_test` reproduz o split de treino do @champion
# (25%, seed 42, estratificado, dados brutos — o XGBoost trata nulos
# nativamente, como na inferência). `verificar_split_do_champion` confirma,
# comparando a AUC recalculada com a registrada no run de treino.
# bias_roi_validation.py importa estas mesmas funções.
# ==============================================================================

from dataclasses import dataclass, field

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config.business_params import CUSTO_APROVAR_MAU_PAGADOR, CUSTO_NEGAR_BOM_PAGADOR

TARGET_COL = "target_dlq_2yrs"
ID_COLS = ["customer_id"]

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

# Split do treino do @champion — não alterar sem retreinar e registrar nova versão
TEST_SIZE = 0.25
SEED = 42

MODEL_SCHEMA = "gold"
MODEL_NAME = "credito_risco_xgb"
MODEL_ALIAS = "champion"

# H4 usa os mesmos nomes de negócio de Q2/Q4 (src/config/business_params.py) —
# perda_media = custo de aprovar um mau pagador; margem_media = custo de negar
# um bom pagador.
PERDA_MEDIA_POR_INADIMPLENCIA_R = CUSTO_APROVAR_MAU_PAGADOR
MARGEM_MEDIA_POR_CLIENTE_BOM_R = CUSTO_NEGAR_BOM_PAGADOR


@dataclass
class TrainedModels:
    xgb_model: object
    logreg_pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred_xgb: np.ndarray
    y_pred_logreg: np.ndarray
    feature_names: list = field(default_factory=list)


# ------------------------------------------------------------------------------
# Dados, split e modelo
# ------------------------------------------------------------------------------
def load_gold_training_data(spark, catalog: str, schema: str = "gold") -> pd.DataFrame:
    """Junta fct_credit_profile + dim_customer. Inclui o lote de scoring
    (alvo nulo); prepare_train_test descarta essas linhas."""
    df_spark = spark.sql(f"""
        SELECT
            f.customer_id, c.age, c.num_dependents,
            f.monthly_income, f.debt_ratio, f.revolving_utilization,
            f.num_open_credit_lines, f.num_real_estate_loans,
            f.num_times_30_59_days_late, f.num_times_60_89_days_late,
            f.num_times_90_days_late, f.total_delinquency_events,
            f.{TARGET_COL}
        FROM {catalog}.{schema}.fct_credit_profile f
        JOIN {catalog}.{schema}.dim_customer c ON f.customer_id = c.customer_id
    """)
    return df_spark.toPandas()


def prepare_train_test(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Split canônico do projeto: só linhas rotuladas, features explícitas,
    dados brutos. Reproduz o split de treino do @champion."""
    faltando = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes nos dados de entrada: {faltando}")

    df = df[df[TARGET_COL].notna() & ~np.isinf(df[TARGET_COL])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def load_champion_model(
    catalog: str,
    schema: str = MODEL_SCHEMA,
    model_name: str = MODEL_NAME,
    model_alias: str = MODEL_ALIAS,
):
    """Carrega o modelo do Registry. Falha alto se não conseguir — a
    validação é do modelo de produção; aqui nunca se treina nem se promove."""
    mlflow.set_registry_uri("databricks-uc")
    uri = f"models:/{catalog}.{schema}.{model_name}@{model_alias}"
    try:
        model = mlflow.xgboost.load_model(uri)
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível carregar {uri}. Rode o notebook de treino/registro antes "
            f"de validar o modelo. Erro original: {e}"
        ) from e
    print(f"Modelo carregado do Registry: {uri}")
    return model


def verificar_split_do_champion(
    catalog: str,
    auc_recalculada: float,
    schema: str = MODEL_SCHEMA,
    model_name: str = MODEL_NAME,
    model_alias: str = MODEL_ALIAS,
    metrica_treino: str = "auc_test",
    tolerancia: float = 1e-6,
) -> float:
    """Compara a AUC recalculada no teste com a registrada no run de treino
    do @champion. Iguais = o teste de agora é o mesmo do treino (sem
    vazamento). Diferentes = o split mudou; falha alto."""
    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    versao = client.get_model_version_by_alias(f"{catalog}.{schema}.{model_name}", model_alias)
    metricas = client.get_run(versao.run_id).data.metrics
    if metrica_treino not in metricas:
        raise RuntimeError(
            f"O run de treino da versão {versao.version} não tem a métrica '{metrica_treino}'; "
            f"não é possível verificar se o split de teste é o mesmo do treino."
        )
    auc_treino = float(metricas[metrica_treino])
    if abs(auc_treino - auc_recalculada) > tolerancia:
        raise RuntimeError(
            f"[FALHA] AUC no teste ({auc_recalculada:.6f}) difere da registrada no treino da "
            f"versão {versao.version} ({auc_treino:.6f}). "
            f"O split de teste não é o mesmo do treino: "
            f"parte dos clientes pode ter sido vista pelo modelo e as métricas sairiam otimistas."
        )
    print(
        f"[OK] Split de teste idêntico ao do treino da versão {versao.version} "
        f"(AUC {auc_recalculada:.6f} = {auc_treino:.6f})."
    )
    return auc_treino


def train_logreg(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED) -> Pipeline:
    """Baseline interpretável do projeto (ver README). A imputação pela
    mediana fica dentro do pipeline, ajustada só no treino."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            ),
        ]
    ).fit(X_train, y_train)


def run_validation(spark, catalog: str, schema: str = MODEL_SCHEMA) -> TrainedModels:
    """Carrega o @champion, reproduz o split do treino (e verifica) e treina
    só a regressão logística de comparação."""
    df = load_gold_training_data(spark, catalog, schema)
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    xgb_model = load_champion_model(catalog, schema)
    y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_pred_xgb)
    verificar_split_do_champion(catalog, auc_xgb, schema)

    logreg_pipeline = train_logreg(X_train, y_train)
    y_pred_logreg = logreg_pipeline.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC XGBoost (@champion): {auc_xgb:.4f}")
    print(f"ROC-AUC Regressão Logística: {roc_auc_score(y_test, y_pred_logreg):.4f}")

    return TrainedModels(
        xgb_model, logreg_pipeline, X_test, y_test, y_pred_xgb, y_pred_logreg, list(FEATURE_COLS)
    )


def compute_shap(models: TrainedModels, sample_size: int = 2000, seed: int = 42):
    """SHAP — base de H1 e H2. Retorna (shap_sample, shap_values, shap_importance_df)."""
    explainer = shap.TreeExplainer(models.xgb_model)
    shap_sample = models.X_test.sample(n=min(sample_size, len(models.X_test)), random_state=seed)
    shap_values = explainer(shap_sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = (
        pd.DataFrame({"feature": models.feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return shap_sample, shap_values, shap_importance_df


def get_logreg_coefficients(models: TrainedModels) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": models.feature_names,
                "coeficiente_logreg": models.logreg_pipeline.named_steps["logreg"].coef_[0],
            }
        )
        .sort_values("coeficiente_logreg", ascending=False)
        .reset_index(drop=True)
    )


def validate_h1(
    shap_sample,
    shap_values,
    shap_importance_df,
    feature="revolving_utilization",
    corr_threshold=0.05,
) -> dict:
    """H1 — utilização de crédito rotativo. Ver README: correlação feature×SHAP.

    Usa Spearman (correlação por postos), não Pearson: a relação esperada é
    monotônica (mais uso do limite -> mais risco), e a utilização tem
    outliers extremos (valores na casa dos milhares) que achatam a
    correlação de Pearson mesmo com a relação claramente crescente. Com
    Pearson, a H1 oscilava entre 0,07 e 0,05 conforme a amostra — em torno
    do próprio critério de decisão."""
    rho = spearmanr(
        shap_sample[feature], shap_values[:, feature].values, nan_policy="omit"
    ).statistic
    corr = float(rho)
    pos = int(shap_importance_df[shap_importance_df["feature"] == feature].index[0] + 1)
    shap_val = float(
        shap_importance_df.loc[shap_importance_df["feature"] == feature, "mean_abs_shap"].values[0]
    )
    status = "Confirmada" if corr > corr_threshold else "Contrariada / Inconclusiva"
    return {
        "status": status,
        "correlacao": corr,
        "posicao_ranking": pos,
        "mean_abs_shap": shap_val,
        "feature": feature,
        "metodo": "Spearman",
    }


def validate_h2(
    shap_importance_df, feature_a="debt_ratio", feature_b="age", relative_margin=0.10
) -> dict:
    """H2 — dívida/renda vs. idade. Ver README: diferença relativa de |SHAP| médio."""
    shap_a = float(
        shap_importance_df.loc[shap_importance_df["feature"] == feature_a, "mean_abs_shap"].values[
            0
        ]
    )
    shap_b = float(
        shap_importance_df.loc[shap_importance_df["feature"] == feature_b, "mean_abs_shap"].values[
            0
        ]
    )
    relative_diff = (shap_a - shap_b) / shap_b if shap_b > 0 else float("inf")
    status = "Confirmada" if relative_diff > relative_margin else "Contrariada / Inconclusiva"
    return {
        "status": status,
        "diferenca_relativa": relative_diff,
        "shap_a": shap_a,
        "shap_b": shap_b,
        "feature_a": feature_a,
        "feature_b": feature_b,
    }


def validate_h3(y_test, y_pred_xgb, min_approval_rate: float = 0.70) -> dict:
    """H3 — threshold ótimo. Ver README: varredura respeitando piso de aprovação."""
    baseline_default_rate = float(y_test.mean())
    thresholds = np.arange(0.05, 0.95, 0.05)
    rows = []
    for t in thresholds:
        approved = y_pred_xgb < t
        approval_rate = float(approved.mean())
        default_rate_approved = float(y_test[approved].mean()) if approved.sum() > 0 else np.nan
        rows.append(
            {
                "threshold": round(float(t), 2),
                "approval_rate": approval_rate,
                "default_rate_approved": default_rate_approved,
            }
        )
    threshold_df = pd.DataFrame(rows)

    viable = threshold_df[threshold_df["approval_rate"] >= min_approval_rate]
    if len(viable) > 0:
        best_row = viable.sort_values("default_rate_approved").iloc[0]
        reduction = baseline_default_rate - best_row["default_rate_approved"]
        status = "Confirmada" if reduction > 0 else "Contrariada / Inconclusiva"
    else:
        best_row, reduction = None, None
        status = "Inconclusiva (nenhum threshold atinge o piso de aprovação definido)"

    return {
        "status": status,
        "threshold_df": threshold_df,
        "best_row": best_row,
        "reduction": reduction,
        "baseline_default_rate": baseline_default_rate,
        "min_approval_rate": min_approval_rate,
    }


def validate_h4(
    y_test,
    y_pred_xgb,
    h3_result: dict,
    perda_media_inadimplencia: float = PERDA_MEDIA_POR_INADIMPLENCIA_R,
    margem_media_bom_pagador: float = MARGEM_MEDIA_POR_CLIENTE_BOM_R,
) -> dict:
    """H4 — impacto financeiro. Ver README: herda o threshold de H3."""
    best_row = h3_result["best_row"]
    if best_row is None:
        return {
            "status": "Não calculada (H3 não encontrou threshold viável)",
            "perda_evitada": None,
            "custo_oportunidade": None,
            "impacto_liquido": None,
            "threshold_usado": None,
        }

    t_final = best_row["threshold"]
    approved = y_pred_xgb < t_final

    maus_pagadores_evitados = int(((y_test == 1) & (~approved)).sum())
    perda_evitada = maus_pagadores_evitados * perda_media_inadimplencia

    bons_pagadores_rejeitados = int(((y_test == 0) & (~approved)).sum())
    custo_oportunidade = bons_pagadores_rejeitados * margem_media_bom_pagador

    impacto_liquido = perda_evitada - custo_oportunidade
    status = "Confirmada" if impacto_liquido > 0 else "Contrariada / Inconclusiva"

    return {
        "status": status,
        "threshold_usado": t_final,
        "maus_pagadores_evitados": maus_pagadores_evitados,
        "perda_evitada": perda_evitada,
        "bons_pagadores_rejeitados": bons_pagadores_rejeitados,
        "custo_oportunidade": custo_oportunidade,
        "impacto_liquido": impacto_liquido,
    }


def build_hypotheses_summary(h1: dict, h2: dict, h3: dict, h4: dict) -> pd.DataFrame:
    best_row = h3["best_row"]
    return pd.DataFrame(
        [
            {
                "hipotese": "H1",
                "descricao": "Clientes com maior revolving_utilization têm maior probabilidade de inadimplência",
                "metrica": h1["correlacao"],
                "status": h1["status"],
                "evidencia": f"Correlação de Spearman SHAP-Feature de {h1['correlacao']:.4f}. Posição {h1['posicao_ranking']}º no ranking SHAP.",
                "limitacao": "Não testa o mecanismo causal ('menor folga financeira'), apenas associação via SHAP.",
            },
            {
                "hipotese": "H2",
                "descricao": "debt_ratio tem poder preditivo maior que age isoladamente",
                "metrica": h2["diferenca_relativa"],
                "status": h2["status"],
                "evidencia": f"debt_ratio SHAP ({h2['shap_a']:.4f}) vs age SHAP ({h2['shap_b']:.4f}). Diferença relativa: {h2['diferenca_relativa']:.2%}.",
                "limitacao": "Compara só com 'age', não com o 'perfil demográfico' como um todo.",
            },
            {
                "hipotese": "H3",
                "descricao": "Existe threshold que reduz inadimplência da carteira aprovada sem violar piso de aprovação",
                "metrica": h3["reduction"] if h3["reduction"] is not None else np.nan,
                "status": h3["status"],
                "evidencia": (
                    f"Threshold {best_row['threshold']}, aprovação {best_row['approval_rate']:.2%}, "
                    f"inadimplência aprovados {best_row['default_rate_approved']:.2%}"
                    if best_row is not None
                    else "Nenhum threshold atingiu o piso de aprovação definido."
                ),
                "limitacao": "Não testa estabilidade do threshold ao longo do tempo (drift).",
            },
            {
                "hipotese": "H4",
                "descricao": "Impacto financeiro líquido de aplicar o modelo (perda evitada vs. custo de oportunidade)",
                "metrica": h4["impacto_liquido"] if h4["impacto_liquido"] is not None else np.nan,
                "status": h4["status"],
                "evidencia": (
                    f"Perda evitada: R$ {h4['perda_evitada']:,.2f} | Custo de oportunidade: R$ {h4['custo_oportunidade']:,.2f}"
                    if h4["impacto_liquido"] is not None
                    else "Dependente de H3."
                ),
                "limitacao": "Valores de perda/margem são ilustrativos — substituir por dados financeiros reais da instituição.",
            },
        ]
    )


def persist_hypotheses_summary(
    spark,
    summary_df: pd.DataFrame,
    catalog: str,
    schema: str = "gold",
    table_name: str = "gold_hypotheses_validation",
) -> None:
    target_table = f"{catalog}.{schema}.{table_name}"
    spark.createDataFrame(summary_df).write.mode("overwrite").format("delta").saveAsTable(
        target_table
    )
    print(f"[OK] Tabela salva: {target_table}")
