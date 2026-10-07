#!/bin/bash
# ver a versao do ruff
# grep -A1 "ruff-pre-commit" .pre-commit-config.yaml
# pip install -q ruff==0.5.0 pre-commit-hooks

# executar da raiz do projeto: bash scripts/setup_terminal.sh    -> rodar via gitbash
pip install -q "ruff==$(grep -A1 'ruff-pre-commit' .pre-commit-config.yaml | grep rev | sed -E 's/.*v([0-9.]+).*/\1/')" pre-commit-hooks


# executar: bash scripts/setup_terminal.sh

# pip install -q ruff==0.8.4 pre-commit-hooks
# pip install -q "ruff==$(grep -A1 'ruff-pre-commit' .pre-commit-config.yaml | grep rev | sed -E 's/.*v([0-9.]+).*/\1/')" pre-commit-hooks

# rodar na maquina local:  uv add --dev ruff==0.5.0    ->  (precisa ser a mesma do databricks e do github)