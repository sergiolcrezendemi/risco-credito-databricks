# src/models/bias_roi_validation.py
# ==============================================================================
# PERGUNTAS Q1-Q4 DO README (distintas das hipóteses H1-H4 — ver
# src/models/hypothesis_validation.py para essas):
#   Q1 - Ranking de importância via SHAP
#   Q2 - Threshold de custo mínimo (custo assimétrico) vs. threshold operacional
#   Q3 - Viés por faixa de idade e renda, no THRESHOLD OPERACIONAL
#   Q3b - Calibração das probabilidades (pré-requisito para precificar)
#   Q3c - XGBoost vs. regressão logística na MESMA taxa de aprovação
#   Q4 - Impacto financeiro estimado (R$) no threshold operacional
#
# DECISÕES DESTA VERSÃO
#   - Threshold operacional único (0,45, definido em H3 pelo piso de
#     aprovação) para viés e impacto financeiro. O threshold de custo mínimo
#     (Q2) é mostrado como comparação, não como régua de decisão: medir viés
#     numa régua que não seria usada responde a uma pergunta que ninguém fez.
#   - Features explícitas (FEATURE_COLS): qualquer coluna nova na Gold (ex.:
#     `_gold_processed_at`) não vira feature por acidente.
#   - O notebook só CARREGA o @champion. A versão anterior treinava e
#     promovia um modelo novo a @champion se o carregamento falhasse por
#     qualquer motivo — um erro transitório de rede podia trocar o modelo
#     de produção em silêncio.
#   - Renda não informada (~20% da base) vira faixa própria. Antes era
#     preenchida com 0 e caía na faixa de menor renda, misturando dois
#     grupos diferentes na análise de viés.
#   - Critério de viés: métricas reconhecidas de fairness (razão de taxa de
#     aprovação — regra dos 4/5 — e taxa de bons pagadores negados com
#     intervalo de confiança) no lugar de "média + 1 desvio-padrão", que
#     com 5 grupos dispara quase sempre, por acaso.
#   - Removidas as checagens H1/H2 de top-5: usavam definições antigas das
#     hipóteses, diferentes das de hypothesis_validation.py.
#   - Checagem de reprodutibilidade do split: a AUC recalculada no teste
#     precisa ser igual à registrada no run de treino do @champion. Se o
#     split mudar (ex.: o Spark devolver as linhas em outra ordem), o teste
#     passaria a conter clientes vistos no treino e as métricas sairiam
#     otimistas — a checagem interrompe a execução antes disso.
#   - Split, carregamento do @champion, checagem do split e regressão
#     logística vêm de hypothesis_validation — definidos num só lugar.
#   - Q3d: experimento sem a variável idade, para medir quanto do viés
#     etário vem da variável em si e quanto chega por outras variáveis.
# ==============================================================================

import logging

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from src.config.business_params import (
    CUSTO_APROVAR_MAU_PAGADOR,
    CUSTO_NEGAR_BOM_PAGADOR,
)
from src.models.hypothesis_validation import (  # noqa: F401 — reexportados para o notebook 02
    FEATURE_COLS,
    TARGET_COL,
    load_champion_model,
    prepare_train_test,
    train_logreg,
    verificar_split_do_champion,
)

logging.getLogger("mlflow").setLevel(logging.ERROR)

# Critérios de viés (pontos de partida documentados — calibrar com risco/jurídico)
LIMIAR_RAZAO_APROVACAO = 0.80  # regra dos 4/5
MIN_BONS_POR_GRUPO = 100  # abaixo disso, a taxa de bons negados é instável demais
Z_95 = 1.96


# ------------------------------------------------------------------------------
# Dados e modelo
# ------------------------------------------------------------------------------
# prepare_train_test, load_champion_model e verificar_split_do_champion vêm de
# hypothesis_validation (importados acima): um único split e um único modelo
# para os dois notebooks de validação.


# ------------------------------------------------------------------------------
# Q1 — SHAP
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# Q2 — Threshold
# ------------------------------------------------------------------------------
def _custo(y_true, y_pred, custo_negar_bom, custo_aprovar_mau) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "custo_total": float(fp * custo_negar_bom + fn * custo_aprovar_mau),
        "taxa_aprovacao": float((tn + fn) / len(y_true)),
    }


def optimize_threshold_asymmetric(
    y_test: pd.Series,
    y_proba_test: np.ndarray,
    threshold_operacional: float,
    custo_negar_bom: float = CUSTO_NEGAR_BOM_PAGADOR,
    custo_aprovar_mau: float = CUSTO_APROVAR_MAU_PAGADOR,
) -> dict:
    """Q2 — varre thresholds e encontra o de custo mínimo, comparando com o
    threshold operacional. A diferença entre os dois é o 'preço' de manter
    o piso de aprovação definido pelo negócio em H3."""
    rows = []
    for t in np.round(np.linspace(0.01, 0.99, 99), 2):
        linha = _custo(y_test, (y_proba_test >= t).astype(int), custo_negar_bom, custo_aprovar_mau)
        rows.append({"threshold": float(t), **linha})

    df_custos = pd.DataFrame(rows)
    melhor = df_custos.loc[df_custos["custo_total"].idxmin()]
    operacional = _custo(
        y_test,
        (y_proba_test >= threshold_operacional).astype(int),
        custo_negar_bom,
        custo_aprovar_mau,
    )
    return {
        "df_custos": df_custos,
        "threshold_custo_minimo": float(melhor["threshold"]),
        "custo_threshold_custo_minimo": float(melhor["custo_total"]),
        "aprovacao_threshold_custo_minimo": float(melhor["taxa_aprovacao"]),
        "threshold_operacional": float(threshold_operacional),
        "custo_threshold_operacional": operacional["custo_total"],
        "aprovacao_threshold_operacional": operacional["taxa_aprovacao"],
        "custo_do_piso_de_aprovacao": operacional["custo_total"] - float(melhor["custo_total"]),
    }


# ------------------------------------------------------------------------------
# Q3 — Viés
# ------------------------------------------------------------------------------
def _intervalo_wilson(sucessos: int, n: int, z: float = Z_95) -> tuple:
    """Intervalo de confiança de Wilson para uma proporção — mais estável
    que o intervalo normal em grupos pequenos ou proporções extremas."""
    if n == 0:
        return (np.nan, np.nan)
    p = sucessos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margem = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centro - margem, centro + margem)


def faixa_idade(idade: pd.Series) -> pd.Series:
    faixas = pd.cut(
        idade,
        bins=[18, 30, 40, 50, 60, np.inf],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
        include_lowest=True,
    )
    return faixas.cat.add_categories("não informada").fillna("não informada")


def faixa_renda(renda: pd.Series, n_faixas: int = 5) -> pd.Series:
    """Quintis calculados só sobre quem informou renda; os demais formam uma
    faixa própria em vez de serem tratados como renda zero."""
    informada = renda.notna()
    faixas = pd.Series("não informada", index=renda.index, dtype="object")
    quintis = pd.qcut(renda[informada], q=n_faixas, duplicates="drop")
    rotulos = [f"Q{i + 1}" for i in range(quintis.cat.categories.size)]
    rotulos[0] += " (menor)"
    rotulos[-1] += " (maior)"
    faixas[informada] = quintis.cat.rename_categories(rotulos).astype(str)
    ordem = rotulos + ["não informada"]
    return pd.Categorical(faixas, categories=ordem, ordered=True)


def _metricas_por_grupo(df_val: pd.DataFrame, coluna_grupo: str) -> pd.DataFrame:
    """Por grupo: tamanho, inadimplência real, aprovação e os dois tipos de
    erro. 'Bons negados' (FPR) é a métrica de justiça central: mede se bons
    pagadores de um grupo pagam mais pelo erro do modelo que os de outro."""
    fpr_geral = df_val.loc[df_val["y_true"] == 0, "y_pred"].mean()
    linhas = []
    for grupo, sub in df_val.groupby(coluna_grupo, observed=True):
        tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1]).ravel()
        n_bons, n_maus = tn + fp, fn + tp
        ic_baixo, ic_alto = _intervalo_wilson(fp, n_bons)
        linhas.append(
            {
                "grupo": str(grupo),
                "n": len(sub),
                "inadimplencia_real": n_maus / len(sub),
                "taxa_aprovacao": (tn + fn) / len(sub),
                "bons_negados": fp / n_bons if n_bons else np.nan,
                "bons_negados_ic95": f"{ic_baixo:.1%}–{ic_alto:.1%}" if n_bons else "",
                "maus_aprovados": fn / n_maus if n_maus else np.nan,
                "amostra_pequena": n_bons < MIN_BONS_POR_GRUPO,
                "_ic_baixo": ic_baixo,
            }
        )
    tabela = pd.DataFrame(linhas)
    tabela["razao_aprovacao"] = tabela["taxa_aprovacao"] / tabela["taxa_aprovacao"].max()
    # Bons negados significativamente acima da média geral: o limite inferior
    # do IC do grupo fica acima da taxa geral
    tabela["bons_negados_acima_da_media"] = tabela["_ic_baixo"] > fpr_geral
    return tabela.drop(columns="_ic_baixo")


def bias_analysis(
    X_test: pd.DataFrame, y_test: pd.Series, y_proba_test: np.ndarray, threshold: float
) -> dict:
    """Q3 — viés por faixa de idade e de renda, no threshold informado
    (use o operacional)."""
    df_val = X_test.copy()
    df_val["y_true"] = y_test.values
    df_val["y_proba"] = y_proba_test
    df_val["y_pred"] = (df_val["y_proba"] >= threshold).astype(int)
    df_val["faixa_idade"] = faixa_idade(df_val["age"])
    df_val["faixa_renda"] = faixa_renda(df_val["monthly_income"])

    tabelas = {
        "idade": _metricas_por_grupo(df_val, "faixa_idade"),
        "renda": _metricas_por_grupo(df_val, "faixa_renda"),
    }

    alertas = []
    for nome, tabela in tabelas.items():
        for _, row in tabela.iterrows():
            ressalva = (
                " (amostra pequena — interpretar com cautela)" if row["amostra_pequena"] else ""
            )
            if row["razao_aprovacao"] < LIMIAR_RAZAO_APROVACAO:
                alertas.append(
                    f"[{nome}] {row['grupo']}: aprovação {row['taxa_aprovacao']:.1%} = "
                    f"{row['razao_aprovacao']:.2f} da maior aprovação "
                    f"(< {LIMIAR_RAZAO_APROVACAO}); "
                    f"inadimplência real do grupo: {row['inadimplencia_real']:.1%}{ressalva}"
                )
            if row["bons_negados_acima_da_media"]:
                alertas.append(
                    f"[{nome}] {row['grupo']}: bons pagadores negados {row['bons_negados']:.1%} "
                    f"(IC95 {row['bons_negados_ic95']}), acima da média geral{ressalva}"
                )

    return {
        "tabela_idade": tabelas["idade"],
        "tabela_renda": tabelas["renda"],
        "alertas": alertas,
        "df_val": df_val,
    }


# ------------------------------------------------------------------------------
# Q3b — Calibração
# ------------------------------------------------------------------------------
def calibration_analysis(
    y_test: pd.Series, y_proba_test: np.ndarray, n_bins: int = 10, seed: int = 42
) -> dict:
    """Diagnostica se a probabilidade prevista corresponde à inadimplência
    observada — necessário para usar o score na precificação, não só na
    ordenação. Ajusta uma calibração isotônica em metade do teste e avalia
    na outra metade (sem avaliar no mesmo dado em que calibrou).

    A calibração isotônica é monotônica: não muda a ordenação dos clientes
    (AUC e decisões de aprovação ficam iguais, com o threshold convertido
    para a nova escala). Só corrige o valor da probabilidade."""
    y = np.asarray(y_test)
    idx_cal, idx_ava = train_test_split(
        np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed
    )
    iso = IsotonicRegression(out_of_bounds="clip").fit(y_proba_test[idx_cal], y[idx_cal])
    p_bruta, p_calibrada, y_ava = (
        y_proba_test[idx_ava],
        iso.predict(y_proba_test[idx_ava]),
        y[idx_ava],
    )

    def _resumo(p):
        obs, prev = calibration_curve(y_ava, p, n_bins=n_bins, strategy="quantile")
        return {
            "brier": float(brier_score_loss(y_ava, p)),
            "media_prevista": float(p.mean()),
            "erro_calibracao_medio": float(np.mean(np.abs(obs - prev))),
            "curva": pd.DataFrame({"prob_prevista": prev, "inadimplencia_observada": obs}),
        }

    return {
        "inadimplencia_observada": float(y_ava.mean()),
        "bruta": _resumo(p_bruta),
        "calibrada": _resumo(p_calibrada),
        "calibrador": iso,
    }


# ------------------------------------------------------------------------------
# Q3c — XGBoost vs. regressão logística na mesma taxa de aprovação
# ------------------------------------------------------------------------------
def _aprovados_na_taxa(proba: np.ndarray, taxa_aprovacao: float) -> np.ndarray:
    """Máscara dos `taxa_aprovacao` clientes de menor risco previsto."""
    n_aprovados = int(round(taxa_aprovacao * len(proba)))
    mascara = np.zeros(len(proba), dtype=bool)
    mascara[np.argsort(proba, kind="stable")[:n_aprovados]] = True
    return mascara


def _carteira_na_aprovacao(y_true: np.ndarray, proba: np.ndarray, taxa_aprovacao: float) -> dict:
    """Aprova os `taxa_aprovacao` clientes de menor risco previsto."""
    mascara = _aprovados_na_taxa(proba, taxa_aprovacao)
    return {
        "inadimplencia_aprovados": float(y_true[mascara].mean()),
        "maus_recusados": int(((~mascara) & (y_true == 1)).sum()),
        "bons_negados": int(((~mascara) & (y_true == 0)).sum()),
    }


def compare_with_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba_xgb: np.ndarray,
    taxa_aprovacao: float,
) -> pd.DataFrame:
    """Compara as duas carteiras aprovando a MESMA proporção de clientes.
    Mede quanto o XGBoost agrega sobre um modelo mais simples, em vez de
    comparar com 'aprovar todo mundo', política que nenhuma instituição usa.

    A logística é a mesma do baseline do projeto (train_logreg, em
    hypothesis_validation)."""
    logistica = train_logreg(X_train, y_train)
    y_proba_lr = logistica.predict_proba(X_test)[:, 1]
    y = np.asarray(y_test)

    linhas = []
    for nome, proba in [("XGBoost (@champion)", y_proba_xgb), ("Regressão logística", y_proba_lr)]:
        linhas.append(
            {
                "modelo": nome,
                "auc": float(roc_auc_score(y, proba)),
                **_carteira_na_aprovacao(y, proba, taxa_aprovacao),
            }
        )
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------------------
# Q3d — Experimento sem a variável idade
# ------------------------------------------------------------------------------
# Hiperparâmetros com que o @champion (versão 1) foi treinado, para que a
# única diferença entre os dois modelos seja a variável removida.
HIPERPARAMETROS_CHAMPION = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
}


def experimento_sem_variavel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba_champion: np.ndarray,
    taxa_aprovacao: float,
    variavel: str = "age",
    seed: int = 42,
) -> dict:
    """Treina um XGBoost igual ao @champion, mas sem `variavel`, e compara os
    dois na MESMA taxa de aprovação: desempenho da carteira e taxa de bons
    pagadores negados por faixa etária.

    Remover a variável não garante remover o viés: outras variáveis podem
    carregar a mesma informação indiretamente (proxies). O experimento mede
    quanto do viés de fato cai e quanto de desempenho se perde."""
    features_sem = [c for c in X_train.columns if c != variavel]
    y_tr = np.asarray(y_train)
    modelo_sem = xgb.XGBClassifier(
        **HIPERPARAMETROS_CHAMPION,
        scale_pos_weight=float((y_tr == 0).sum() / (y_tr == 1).sum()),
        random_state=seed,
    ).fit(X_train[features_sem], y_train)
    y_proba_sem = modelo_sem.predict_proba(X_test[features_sem])[:, 1]

    y = np.asarray(y_test)
    faixas = pd.Series(faixa_idade(X_test["age"]).astype(str), index=X_test.index)
    modelos = {"com idade (@champion)": y_proba_champion, f"sem {variavel}": y_proba_sem}

    resumo, por_faixa = [], {}
    for nome, proba in modelos.items():
        recusado = ~_aprovados_na_taxa(proba, taxa_aprovacao)
        bons = y == 0
        taxas = (
            pd.DataFrame({"faixa": faixas.values, "recusado": recusado, "bom": bons})
            .query("bom")
            .groupby("faixa", sort=False)["recusado"]
            .mean()
        )
        por_faixa[nome] = taxas
        resumo.append(
            {
                "modelo": nome,
                "auc": float(roc_auc_score(y, proba)),
                **_carteira_na_aprovacao(y, proba, taxa_aprovacao),
                "razao_bons_negados_18_30_vs_60_mais": float(taxas["18-30"] / taxas["60+"]),
            }
        )

    ordem = ["18-30", "31-40", "41-50", "51-60", "60+", "não informada"]
    tabela_faixas = pd.DataFrame(por_faixa).reindex([f for f in ordem if f in faixas.unique()])
    tabela_faixas.index.name = "faixa_idade"
    nomes = list(modelos)
    tabela_faixas["variacao_pp"] = (tabela_faixas[nomes[1]] - tabela_faixas[nomes[0]]) * 100

    return {
        "resumo": pd.DataFrame(resumo),
        "bons_negados_por_faixa": tabela_faixas.reset_index(),
        "y_proba_sem": y_proba_sem,
    }


# ------------------------------------------------------------------------------
# Q4 — Impacto financeiro
# ------------------------------------------------------------------------------
def financial_impact(
    df_val: pd.DataFrame,
    custo_aprovar_mau: float = CUSTO_APROVAR_MAU_PAGADOR,
    custo_negar_bom: float = CUSTO_NEGAR_BOM_PAGADOR,
) -> dict:
    """Q4 — perda evitada (maus recusados) vs. custo de oportunidade (bons
    negados), com os MESMOS custos usados em Q2. Inclui o ponto de
    equilíbrio: a razão perda/margem a partir da qual o modelo se paga —
    uma resposta que não depende dos valores ilustrativos."""
    maus_recusados = int(((df_val["y_pred"] == 1) & (df_val["y_true"] == 1)).sum())
    bons_negados = int(((df_val["y_pred"] == 1) & (df_val["y_true"] == 0)).sum())
    aprovados = df_val[df_val["y_pred"] == 0]

    perda_evitada = maus_recusados * custo_aprovar_mau
    custo_oportunidade = bons_negados * custo_negar_bom
    return {
        "n_operacoes": len(df_val),
        "taxa_aprovacao": len(aprovados) / len(df_val),
        "taxa_sem_modelo": float(df_val["y_true"].mean()),
        "taxa_com_modelo": float(aprovados["y_true"].mean()) if len(aprovados) else 0.0,
        "maus_recusados": maus_recusados,
        "bons_negados": bons_negados,
        "perda_evitada": perda_evitada,
        "custo_oportunidade": custo_oportunidade,
        "impacto_liquido": perda_evitada - custo_oportunidade,
        "razao_equilibrio_perda_margem": bons_negados / maus_recusados
        if maus_recusados
        else np.nan,
    }


# ------------------------------------------------------------------------------
# Registro
# ------------------------------------------------------------------------------
def log_validation_to_mlflow(
    threshold_result: dict,
    financial_result: dict,
    ranking_shap: pd.DataFrame,
    bias_result: dict,
    calibration_result: dict,
    comparison: pd.DataFrame,
    experimento_sem_idade: dict | None = None,
) -> None:
    with mlflow.start_run(run_name="validacao_shap_threshold_vies_roi"):
        mlflow.log_params(
            {
                "threshold_operacional": threshold_result["threshold_operacional"],
                "threshold_custo_minimo": threshold_result["threshold_custo_minimo"],
                "custo_aprovar_mau_pagador": CUSTO_APROVAR_MAU_PAGADOR,
                "custo_negar_bom_pagador": CUSTO_NEGAR_BOM_PAGADOR,
                "limiar_razao_aprovacao": LIMIAR_RAZAO_APROVACAO,
            }
        )
        mlflow.log_metrics(
            {
                "custo_do_piso_de_aprovacao": threshold_result["custo_do_piso_de_aprovacao"],
                "impacto_financeiro_liquido": financial_result["impacto_liquido"],
                "razao_equilibrio_perda_margem": financial_result["razao_equilibrio_perda_margem"],
                "taxa_inadimplencia_sem_modelo": financial_result["taxa_sem_modelo"],
                "taxa_inadimplencia_com_modelo": financial_result["taxa_com_modelo"],
                "qtd_alertas_vies": len(bias_result["alertas"]),
                "brier_bruto": calibration_result["bruta"]["brier"],
                "brier_calibrado": calibration_result["calibrada"]["brier"],
                "erro_calibracao_bruto": calibration_result["bruta"]["erro_calibracao_medio"],
                "erro_calibracao_calibrado": calibration_result["calibrada"][
                    "erro_calibracao_medio"
                ],
            }
        )
        mlflow.log_dict(ranking_shap.to_dict(orient="records"), "shap_ranking.json")
        mlflow.log_dict(
            bias_result["tabela_idade"].to_dict(orient="records"), "vies_por_idade.json"
        )
        mlflow.log_dict(
            bias_result["tabela_renda"].to_dict(orient="records"), "vies_por_renda.json"
        )
        mlflow.log_dict({"alertas": bias_result["alertas"]}, "alertas_vies.json")
        mlflow.log_dict(comparison.to_dict(orient="records"), "comparacao_logistica.json")
        if experimento_sem_idade is not None:
            mlflow.log_dict(
                experimento_sem_idade["resumo"].to_dict(orient="records"),
                "experimento_sem_idade_resumo.json",
            )
            mlflow.log_dict(
                experimento_sem_idade["bons_negados_por_faixa"].to_dict(orient="records"),
                "experimento_sem_idade_por_faixa.json",
            )
    print("\nValidação concluída e registrada no MLflow.")
