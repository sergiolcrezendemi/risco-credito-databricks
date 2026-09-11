# src/models/hypothesis_validation.py
# ==============================================================================
# VALIDAÇÃO DAS HIPÓTESES H1-H4 (ver README.md)
#
# Lê da GOLD (fct_credit_profile + dim_customer) — não da Silver — para ficar
# consistente com o notebook irmão (03_analise_shap_threshold_vies_roi.py),
# que já lê de lá. Os valores financeiros (H4) e o catálogo não ficam mais
# hardcoded: vêm de parâmetro, com um único default documentado aqui, para
# não divergir entre os dois notebooks de novo.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config.business_params import CUSTO_APROVAR_MAU_PAGADOR, CUSTO_NEGAR_BOM_PAGADOR

TARGET_COL = "target_dlq_2yrs"
ID_COLS = ["customer_id"]

# H4 usa os mesmos nomes de negócio de Q2/Q4 (src/config/business_params.py) —
# perda_media = custo de aprovar um mau pagador; margem_media = custo de negar
# um bom pagador. Antes eram duas constantes com valores diferentes sem motivo.
PERDA_MEDIA_POR_INADIMPLENCIA_R = CUSTO_APROVAR_MAU_PAGADOR
MARGEM_MEDIA_POR_CLIENTE_BOM_R = CUSTO_NEGAR_BOM_PAGADOR


@dataclass
class TrainedModels:
    xgb_model: XGBClassifier
    logreg_pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred_xgb: np.ndarray
    y_pred_logreg: np.ndarray
    feature_names: list = field(default_factory=list)


def load_gold_training_data(spark, catalog: str, schema: str = "gold") -> pd.DataFrame:
    """Junta fct_credit_profile + dim_customer — mesma junção que o
    notebook de viés/ROI já usa."""
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


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.20, seed: int = 42):
    """Descarta linhas sem target, faz split estratificado e imputa nulos
    (mediana, fit só no treino — sem vazamento)."""
    target_invalido = df[TARGET_COL].isna() | np.isinf(df[TARGET_COL])
    df_valid = df[~target_invalido].copy()

    feature_cols = [c for c in df_valid.columns if c not in ID_COLS + [TARGET_COL]]
    X = df_valid[feature_cols]
    y = df_valid[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[("impute_median", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    names = list(preprocessor.get_feature_names_out())

    X_train_df = pd.DataFrame(X_train_prep, columns=names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_prep, columns=names, index=X_test.index)
    return X_train_df, X_test_df, y_train, y_test, names


def train_models(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> tuple:
    """XGBoost (principal) + Regressão Logística (baseline interpretável) — ver README."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=float(neg / pos), subsample=0.8, colsample_bytree=0.8,
        random_state=seed, eval_metric="auc",
    )
    xgb_model.fit(X_train, y_train)

    logreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])
    logreg_pipeline.fit(X_train, y_train)
    return xgb_model, logreg_pipeline


def run_training(spark, catalog: str, schema: str = "gold", seed: int = 42) -> TrainedModels:
    df = load_gold_training_data(spark, catalog, schema)
    X_train, X_test, y_train, y_test, feature_names = prepare_train_test(df, seed=seed)
    xgb_model, logreg_pipeline = train_models(X_train, y_train, seed=seed)

    y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_logreg = logreg_pipeline.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC XGBoost: {roc_auc_score(y_test, y_pred_xgb):.4f}")
    print(f"ROC-AUC Regressão Logística: {roc_auc_score(y_test, y_pred_logreg):.4f}")

    return TrainedModels(xgb_model, logreg_pipeline, X_test, y_test, y_pred_xgb, y_pred_logreg, feature_names)


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
    return pd.DataFrame({
        "feature": models.feature_names,
        "coeficiente_logreg": models.logreg_pipeline.named_steps["logreg"].coef_[0],
    }).sort_values("coeficiente_logreg", ascending=False).reset_index(drop=True)


def validate_h1(shap_sample, shap_values, shap_importance_df, feature="revolving_utilization", corr_threshold=0.05) -> dict:
    """H1 — utilização de crédito rotativo. Ver README: correlação feature×SHAP."""
    corr = float(np.corrcoef(shap_sample[feature], shap_values[:, feature].values)[0, 1])
    pos = int(shap_importance_df[shap_importance_df["feature"] == feature].index[0] + 1)
    shap_val = float(shap_importance_df.loc[shap_importance_df["feature"] == feature, "mean_abs_shap"].values[0])
    status = "Confirmada" if corr > corr_threshold else "Contrariada / Inconclusiva"
    return {"status": status, "correlacao": corr, "posicao_ranking": pos, "mean_abs_shap": shap_val, "feature": feature}


def validate_h2(shap_importance_df, feature_a="debt_ratio", feature_b="age", relative_margin=0.10) -> dict:
    """H2 — dívida/renda vs. idade. Ver README: diferença relativa de |SHAP| médio."""
    shap_a = float(shap_importance_df.loc[shap_importance_df["feature"] == feature_a, "mean_abs_shap"].values[0])
    shap_b = float(shap_importance_df.loc[shap_importance_df["feature"] == feature_b, "mean_abs_shap"].values[0])
    relative_diff = (shap_a - shap_b) / shap_b if shap_b > 0 else float("inf")
    status = "Confirmada" if relative_diff > relative_margin else "Contrariada / Inconclusiva"
    return {"status": status, "diferenca_relativa": relative_diff, "shap_a": shap_a, "shap_b": shap_b,
            "feature_a": feature_a, "feature_b": feature_b}


def validate_h3(y_test, y_pred_xgb, min_approval_rate: float = 0.70) -> dict:
    """H3 — threshold ótimo. Ver README: varredura respeitando piso de aprovação."""
    baseline_default_rate = float(y_test.mean())
    thresholds = np.arange(0.05, 0.95, 0.05)
    rows = []
    for t in thresholds:
        approved = y_pred_xgb < t
        approval_rate = float(approved.mean())
        default_rate_approved = float(y_test[approved].mean()) if approved.sum() > 0 else np.nan
        rows.append({"threshold": round(float(t), 2), "approval_rate": approval_rate,
                      "default_rate_approved": default_rate_approved})
    threshold_df = pd.DataFrame(rows)

    viable = threshold_df[threshold_df["approval_rate"] >= min_approval_rate]
    if len(viable) > 0:
        best_row = viable.sort_values("default_rate_approved").iloc[0]
        reduction = baseline_default_rate - best_row["default_rate_approved"]
        status = "Confirmada" if reduction > 0 else "Contrariada / Inconclusiva"
    else:
        best_row, reduction = None, None
        status = "Inconclusiva (nenhum threshold atinge o piso de aprovação definido)"

    return {"status": status, "threshold_df": threshold_df, "best_row": best_row,
            "reduction": reduction, "baseline_default_rate": baseline_default_rate,
            "min_approval_rate": min_approval_rate}


def validate_h4(
    y_test, y_pred_xgb, h3_result: dict,
    perda_media_inadimplencia: float = PERDA_MEDIA_POR_INADIMPLENCIA_R,
    margem_media_bom_pagador: float = MARGEM_MEDIA_POR_CLIENTE_BOM_R,
) -> dict:
    """H4 — impacto financeiro. Ver README: herda o threshold de H3."""
    best_row = h3_result["best_row"]
    if best_row is None:
        return {"status": "Não calculada (H3 não encontrou threshold viável)",
                "perda_evitada": None, "custo_oportunidade": None, "impacto_liquido": None,
                "threshold_usado": None}

    t_final = best_row["threshold"]
    approved = y_pred_xgb < t_final

    maus_pagadores_evitados = int(((y_test == 1) & (~approved)).sum())
    perda_evitada = maus_pagadores_evitados * perda_media_inadimplencia

    bons_pagadores_rejeitados = int(((y_test == 0) & (~approved)).sum())
    custo_oportunidade = bons_pagadores_rejeitados * margem_media_bom_pagador

    impacto_liquido = perda_evitada - custo_oportunidade
    status = "Confirmada" if impacto_liquido > 0 else "Contrariada / Inconclusiva"

    return {"status": status, "threshold_usado": t_final,
            "maus_pagadores_evitados": maus_pagadores_evitados, "perda_evitada": perda_evitada,
            "bons_pagadores_rejeitados": bons_pagadores_rejeitados, "custo_oportunidade": custo_oportunidade,
            "impacto_liquido": impacto_liquido}


def build_hypotheses_summary(h1: dict, h2: dict, h3: dict, h4: dict) -> pd.DataFrame:
    best_row = h3["best_row"]
    return pd.DataFrame([
        {
            "hipotese": "H1",
            "descricao": "Clientes com maior revolving_utilization têm maior probabilidade de inadimplência",
            "metrica": h1["correlacao"], "status": h1["status"],
            "evidencia": f"Correlação SHAP-Feature de {h1['correlacao']:.4f}. Posição {h1['posicao_ranking']}º no ranking SHAP.",
            "limitacao": "Não testa o mecanismo causal ('menor folga financeira'), apenas associação via SHAP.",
        },
        {
            "hipotese": "H2",
            "descricao": "debt_ratio tem poder preditivo maior que age isoladamente",
            "metrica": h2["diferenca_relativa"], "status": h2["status"],
            "evidencia": f"debt_ratio SHAP ({h2['shap_a']:.4f}) vs age SHAP ({h2['shap_b']:.4f}). Diferença relativa: {h2['diferenca_relativa']:.2%}.",
            "limitacao": "Compara só com 'age', não com o 'perfil demográfico' como um todo.",
        },
        {
            "hipotese": "H3",
            "descricao": "Existe threshold que reduz inadimplência da carteira aprovada sem violar piso de aprovação",
            "metrica": h3["reduction"] if h3["reduction"] is not None else np.nan, "status": h3["status"],
            "evidencia": (
                f"Threshold {best_row['threshold']}, aprovação {best_row['approval_rate']:.2%}, "
                f"inadimplência aprovados {best_row['default_rate_approved']:.2%}"
                if best_row is not None else "Nenhum threshold atingiu o piso de aprovação definido."
            ),
            "limitacao": "Não testa estabilidade do threshold ao longo do tempo (drift).",
        },
        {
            "hipotese": "H4",
            "descricao": "Impacto financeiro líquido de aplicar o modelo (perda evitada vs. custo de oportunidade)",
            "metrica": h4["impacto_liquido"] if h4["impacto_liquido"] is not None else np.nan, "status": h4["status"],
            "evidencia": (
                f"Perda evitada: R$ {h4['perda_evitada']:,.2f} | Custo de oportunidade: R$ {h4['custo_oportunidade']:,.2f}"
                if h4["impacto_liquido"] is not None else "Dependente de H3."
            ),
            "limitacao": "Valores de perda/margem são ilustrativos — substituir por dados financeiros reais da Mezzo.",
        },
    ])


def persist_hypotheses_summary(spark, summary_df: pd.DataFrame, catalog: str, schema: str = "gold",
                                table_name: str = "gold_hypotheses_validation") -> None:
    target_table = f"{catalog}.{schema}.{table_name}"
    spark.createDataFrame(summary_df).write.mode("overwrite").format("delta").saveAsTable(target_table)
    print(f"[OK] Tabela salva: {target_table}")
