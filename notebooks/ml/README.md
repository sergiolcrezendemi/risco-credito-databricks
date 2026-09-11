    
# notebooks/ml/

Notebooks que dependem de um modelo treinado (XGBoost). Dois tipos aqui,
propositalmente juntos por enquanto (ver decisão em docs/roteiro-execucao.md):

- **Relatórios de validação** (leem, não fazem parte do Job agendado):
  `01_hipoteses_h1_h4.py`, `02_vies_threshold_roi.py`
- **Produção** (roda como task do Job): `04_treino_mlflow.py`

Ordem geral de execução: ver docs/roteiro-execucao.md.