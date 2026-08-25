# Arquitetura — Bronze / Silver / Gold

```
Bronze  -> ingestao bruta (Autoloader ou notebook)
Silver  -> limpeza, padronizacao, tratamento de qualidade
Gold    -> tabela de negocio / feature table, pronta para consumo
```

Ambientes: cada camada existe em tres catalogos Unity Catalog — `_dev`, `_hml`, `_prd` (ver ESTRUTURA-DATABRICKS-GITHUB.md na raiz do pacote).
