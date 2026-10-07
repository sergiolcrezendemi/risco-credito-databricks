#!/bin/bash
set -e

# vai para a raiz do projeto, onde quer que o script seja chamado
# executar:  bash scripts/formatar.sh
cd "$(dirname "$0")/.."

bash scripts/setup_terminal.sh

ruff format notebooks/ src/ tests/
ruff check --fix notebooks/ src/ tests/

# arquivos de texto validados pelo CI, fora de .venv, .git e pastas de rascunho
lista_arquivos() {
  find . -type f \
    \( -name "*.py" -o -name "*.ipynb" \) \
    -not -path "./.venv/*" -not -path "./.git/*" -not -path "*/.temp/*" \
    -not -path "*/.temp_del/*" -not -path "./.ruff_cache/*" \
    -not -path "./.databricks/*" -not -path "./docs/*" -print0
}


# 1ª passada: corrige (os fixers retornam 1 quando alteram algo)
lista_arquivos | xargs -0 trailing-whitespace-fixer || true
lista_arquivos | xargs -0 end-of-file-fixer || true

# 2ª passada: precisa passar limpa
lista_arquivos | xargs -0 trailing-whitespace-fixer
lista_arquivos | xargs -0 end-of-file-fixer

check-yaml databricks.yml resources/*.yml

echo "Pronto. Agora faça commit e push pela interface do Git folder."
