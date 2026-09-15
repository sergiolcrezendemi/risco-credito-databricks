# src/config/business_params.py
# ==============================================================================
# PARÂMETROS FINANCEIROS — fonte única de verdade.
#
# Usados por src/models/hypothesis_validation.py (H4) e
# src/models/bias_roi_validation.py (Q2/Q4). Antes cada módulo declarava a
# própria constante com valores diferentes (8.000/1.200 vs. 15.000/1.500) —
# sem justificativa de negócio, só porque foram escritos em dois lugares
# separados. Agora os dois importam daqui.
#
# São valores ILUSTRATIVOS. Para uso real, substituir pelos valores
# financeiros reais da Mezzo (perda média por inadimplência, margem média
# por cliente aprovado, ticket médio de operação).
#
# H4 e Q4 ainda podem dar números diferentes mesmo com as mesmas constantes
# aqui — não é bug: H4 usa o threshold de H3 (o menor que respeita um piso
# de aprovação de 70%), Q4 usa o threshold de Q2 (o que minimiza custo
# esperado). São respostas a perguntas diferentes; a diferença no threshold
# é intencional, só a constante de custo em R$ não deveria divergir.
# ==============================================================================

CUSTO_APROVAR_MAU_PAGADOR = 15000.0   # perda média ao conceder crédito a um mau pagador
CUSTO_NEGAR_BOM_PAGADOR = 1500.0      # margem perdida ao negar crédito a um bom pagador
VALOR_MEDIO_OPERACAO = 12000.0        # ticket médio de uma operação de crédito (usado só em Q4)
