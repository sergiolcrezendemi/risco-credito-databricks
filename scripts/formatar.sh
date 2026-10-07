#!/bin/bash
set -e

# export PATH="$(python3 -c 'import site; print(site.USER_BASE)')/bin:$PATH"
export PATH="/databricks/python/bin:$PATH"



# executar:  bash scripts/formatar.sh
# vai para a raiz do projeto, onde quer que o script seja chamado
cd "$(dirname "$0")/.."

bash scripts/setup_terminal.sh

# formatação estrita só no código de verdade
ruff format src/ tests/

# lint em tudo, inclusive notebooks
ruff check --fix notebooks/ src/ tests/

# notebooks: só avisa, não altera nem bloqueia
ruff format --check notebooks/ || echo "Aviso: notebooks com formatação diferente (não bloqueia o CI)"

# .py e .ipynb validados pelo CI, fora de .venv, .git e pastas de rascunho
lista_arquivos() {
  find . -type f \
    \( -name "*.py" -o -name "*.ipynb" \) \
    -not -path "./.venv/*" -not -path "./.git/*" -not -path "*/.temp/*" \
    -not -path "*/.temp_del/*" -not -path "./.ruff_cache/*" \
    -not -path "./.databricks/*" -not -path "./docs/*" \
    -not -path "./notebooks/*" -print0
}

# 1ª passada: corrige (os fixers retornam 1 quando alteram algo)
lista_arquivos | xargs -0 trailing-whitespace-fixer || true
lista_arquivos | xargs -0 end-of-file-fixer || true

# 2ª passada: precisa passar limpa 
lista_arquivos | xargs -0 trailing-whitespace-fixer
lista_arquivos | xargs -0 end-of-file-fixer

# valida a sintaxe de todos os YAMLs do projeto
find . -type f \( -name "*.yml" -o -name "*.yaml" \) \
  -not -path "./.venv/*" -not -path "./.git/*" -not -path "*/.temp/*" \
  -not -path "*/.temp_del/*" -not -path "./.databricks/*" -print0 \
  | xargs -0 -r check-yaml

echo "Pronto. Agora faça commit e push pela interface do Git folder."