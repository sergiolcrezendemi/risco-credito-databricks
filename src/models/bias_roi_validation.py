# src/models/bias_roi_validation.py
# ==============================================================================
# PERGUNTAS Q1-Q4 DO README (distintas das hipóteses H1-H4 — ver
# src/models/hypothesis_validation.py para essas):
#   Q1 - Ranking de importância via SHAP (checagem rápida: H1/H2 aparecem no top-5?)
#   Q2 - Threshold ótimo sob CUSTO ASSIMÉTRICO (não só piso de aprovação)
#   Q3 - Estabilidade/viés do modelo por faixa de idade e renda
#   Q4 - Impacto financeiro estimado (R$) vs. política atual
#
# Os parâmetros financeiros vêm de src/config/business_params.py — mesma
# fonte usada por H4 em hypothesis_validation.py, para as duas contas nunca
# mais divergirem por constantes diferentes escritas em dois lugares. H4 e
# Q4 ainda podem dar números diferentes mesmo assim: H4 usa o threshold de
# H3 (piso de aprovação), Q4 usa o threshold de Q2 (custo mínimo) — isso é
# intencional, respondem perguntas diferentes.
# ==============================================================================

import logging

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from mlflow.models.signature import infer_signature
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.hypothesis_validation import TARGET_COL
from src.config.business_params import (
    CUSTO_APROVAR_MAU_PAGADOR,
    CUSTO_NEGAR_BOM_PAGADOR,
    VALOR_MEDIO_OPERACAO,
)

logging.getLogger("mlflow").setLevel(logging.ERROR)

H1_FEATURES = ["num_times_30_59_days_late", "num_times_90_days_late"]
H2_FEATURES = ["revolving_utilization", "monthly_income"]


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    feature_cols = [c for c in df.columns if c not in ["customer_id", TARGET_COL]]
    X, y = df[feature_cols], df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def load_or_train_model(
    spark, catalog: str, schema: str, model_name: str, model_alias: str,
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    seed: int = 42,
):
    """Carrega do Model Registry (Unity Catalog); se o alias ainda não existir,
    treina um baseline, registra e aponta o alias para a nova versão."""
    mlflow.set_registry_uri("databricks-uc")
    full_model_name = f"{catalog}.{schema}.{model_name}"

    try:
        model = mlflow.xgboost.load_model(f"models:/{full_model_name}@{model_alias}")
        print(f"Modelo carregado do Registry: models:/{full_model_name}@{model_alias}")
        return model
    except Exception:
        print(f"Alias '@{model_alias}' ainda não existe para '{full_model_name}'. Treinando baseline...")

    with mlflow.start_run(run_name="baseline_xgb_validacao"):
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=seed,
        )
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("auc_test", auc)

        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
        mlflow.xgboost.log_model(
            model, "model", registered_model_name=full_model_name,
            signature=signature, input_example=X_train.head(5),
        )
        print(f"AUC baseline no teste: {auc:.4f}")

        try:
            client = mlflow.MlflowClient()
            versoes = client.search_model_versions(f"name='{full_model_name}'")
            ultima_versao = max(int(v.version) for v in versoes)
            client.set_registered_model_alias(name=full_model_name, alias=model_alias, version=ultima_versao)
            print(f"Alias '@{model_alias}' apontado para a versão {ultima_versao}.")
        except Exception as e:
            print(f"[AVISO] Não foi possível setar o alias automaticamente: {e}")

    return model


def compute_shap_ranking(model, X_test: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking, shap_values


def check_top_n(features_esperadas: list, top_features: set, top_n: int = 5) -> str:
    presentes = [f for f in features_esperadas if f in top_features]
    if len(presentes) == len(features_esperadas):
        return "CONFIRMADA"
    return "PARCIALMENTE CONFIRMADA" if presentes else "CONTRARIADA"


def optimize_threshold_asymmetric(
    y_test: pd.Series, y_proba_test: np.ndarray,
    custo_negar_bom: float = CUSTO_NEGAR_BOM_PAGADOR,
    custo_aprovar_mau: float = CUSTO_APROVAR_MAU_PAGADOR,
) -> dict:
    """Q2 — varre thresholds minimizando custo esperado (FP × custo_negar_bom
    + FN × custo_aprovar_mau), em vez de só respeitar um piso de aprovação."""
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        y_pred = (y_proba_test >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        custo_total = fp * custo_negar_bom + fn * custo_aprovar_mau
        rows.append({"threshold": t, "fp": fp, "fn": fn, "tp": tp, "tn": tn, "custo_total": custo_total})

    df_custos = pd.DataFrame(rows)
    melhor = df_custos.loc[df_custos["custo_total"].idxmin()]
    custo_05 = df_custos.loc[(df_custos["threshold"] - 0.50).abs().idxmin()]

    return {
        "df_custos": df_custos,
        "threshold_otimo": float(melhor["threshold"]),
        "custo_threshold_otimo": float(melhor["custo_total"]),
        "custo_threshold_050": float(custo_05["custo_total"]),
        "economia": float(custo_05["custo_total"] - melhor["custo_total"]),
        "fp_otimo": int(melhor["fp"]),
        "fn_otimo": int(melhor["fn"]),
    }


def _metricas_por_grupo(df_val: pd.DataFrame, coluna_grupo: str) -> pd.DataFrame:
    linhas = []
    for grupo, sub in df_val.groupby(coluna_grupo, observed=True):
        tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        acc = (tp + tn) / len(sub) if len(sub) > 0 else np.nan
        linhas.append({"grupo": grupo, "n": len(sub), "acuracia": acc,
                        "fpr_bons_negados": fpr, "fnr_maus_aprovados": fnr})
    return pd.DataFrame(linhas)


def bias_analysis(X_test: pd.DataFrame, y_test: pd.Series, y_proba_test: np.ndarray, threshold: float) -> dict:
    """Q3 — performance por faixa de idade e de renda, com flag de
    desproporcionalidade (métrica do grupo > média + 1 desvio-padrão)."""
    df_val = X_test.copy()
    df_val["y_true"] = y_test.values
    df_val["y_proba"] = y_proba_test
    df_val["y_pred"] = (df_val["y_proba"] >= threshold).astype(int)

    df_val["faixa_idade"] = pd.cut(
        df_val["age"], bins=[18, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    )

    renda_labels_padrao = ["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"]
    faixa_renda = pd.qcut(df_val["monthly_income"].fillna(0), q=5, duplicates="drop")
    aviso_renda = None
    if faixa_renda.cat.categories.size == len(renda_labels_padrao):
        faixa_renda = faixa_renda.cat.rename_categories(renda_labels_padrao)
    else:
        aviso_renda = (
            f"faixa_renda: apenas {faixa_renda.cat.categories.size} faixas distintas geradas "
            f"(esperado 5) — dados de 'monthly_income' concentrados/nulos."
        )
    df_val["faixa_renda"] = faixa_renda

    tabela_idade = _metricas_por_grupo(df_val, "faixa_idade")
    tabela_renda = _metricas_por_grupo(df_val, "faixa_renda")

    alertas = []
    for nome, tabela in [("idade", tabela_idade), ("renda", tabela_renda)]:
        for metrica in ["fpr_bons_negados", "fnr_maus_aprovados"]:
            media, desvio = tabela[metrica].mean(), tabela[metrica].std()
            outliers = tabela[tabela[metrica] > media + desvio]
            for _, row in outliers.iterrows():
                alertas.append(f"[{nome}] grupo {row['grupo']}: {metrica} = {row[metrica]:.2%} (desproporcional)")

    return {
        "tabela_idade": tabela_idade, "tabela_renda": tabela_renda,
        "alertas": alertas, "aviso_renda": aviso_renda, "df_val": df_val,
    }


def financial_impact(
    df_val: pd.DataFrame,
    custo_negar_bom: float = CUSTO_NEGAR_BOM_PAGADOR,
    valor_medio_operacao: float = VALOR_MEDIO_OPERACAO,
    taxa_inadimplencia_politica_atual: float = None,
) -> dict:
    """Q4 — perda sem modelo (aprova todos) vs. com modelo (threshold ótimo)."""
    n_operacoes = len(df_val)
    taxa_sem_modelo = (
        taxa_inadimplencia_politica_atual
        if taxa_inadimplencia_politica_atual is not None
        else df_val["y_true"].mean()
    )
    perda_sem_modelo = n_operacoes * taxa_sem_modelo * valor_medio_operacao

    aprovados = df_val[df_val["y_pred"] == 0]
    n_aprovados = len(aprovados)
    taxa_com_modelo = aprovados["y_true"].mean() if n_aprovados > 0 else 0.0
    perda_com_modelo = n_aprovados * taxa_com_modelo * valor_medio_operacao

    bons_negados = df_val[(df_val["y_pred"] == 1) & (df_val["y_true"] == 0)]
    custo_oportunidade = len(bons_negados) * custo_negar_bom

    reducao_bruta = perda_sem_modelo - perda_com_modelo
    impacto_liquido = reducao_bruta - custo_oportunidade

    return {
        "n_operacoes": n_operacoes, "taxa_sem_modelo": taxa_sem_modelo, "perda_sem_modelo": perda_sem_modelo,
        "taxa_com_modelo": taxa_com_modelo, "perda_com_modelo": perda_com_modelo,
        "reducao_bruta": reducao_bruta, "custo_oportunidade": custo_oportunidade,
        "impacto_liquido": impacto_liquido,
    }


def log_validation_to_mlflow(threshold_result: dict, financial_result: dict, ranking_shap: pd.DataFrame,
                              bias_result: dict, status_h1: str, status_h2: str) -> None:
    with mlflow.start_run(run_name="validacao_shap_threshold_vies_roi"):
        mlflow.log_param("threshold_otimo", threshold_result["threshold_otimo"])
        mlflow.log_param("custo_aprovar_mau_pagador", CUSTO_APROVAR_MAU_PAGADOR)
        mlflow.log_param("custo_negar_bom_pagador", CUSTO_NEGAR_BOM_PAGADOR)
        mlflow.log_metric("custo_total_threshold_otimo", threshold_result["custo_threshold_otimo"])
        mlflow.log_metric("impacto_financeiro_liquido", financial_result["impacto_liquido"])
        mlflow.log_metric("taxa_inadimplencia_sem_modelo", financial_result["taxa_sem_modelo"])
        mlflow.log_metric("taxa_inadimplencia_com_modelo", financial_result["taxa_com_modelo"])
        mlflow.log_dict(ranking_shap.to_dict(orient="records"), "shap_ranking.json")
        mlflow.log_dict(bias_result["tabela_idade"].to_dict(orient="records"), "vies_por_idade.json")
        mlflow.log_dict(bias_result["tabela_renda"].to_dict(orient="records"), "vies_por_renda.json")
        mlflow.log_param("h1_status", status_h1)
        mlflow.log_param("h2_status", status_h2)
    print("\nValidação concluída e registrada no MLflow.")
