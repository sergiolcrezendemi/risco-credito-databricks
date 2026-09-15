#!/usr/bin/env python3
# scripts/check_notebook_hygiene.py
# ==============================================================================
# Bloqueia commit de notebooks com comandos que só deveriam existir em sessão
# interativa de dev. Comentário explicando "isso só roda em dev" NÃO conta
# como proteção — o Databricks executa a linha do mesmo jeito seja Job ou
# manual. A única forma confiável de garantir que algo não rode em produção
# é o código não estar no arquivo. Este check substitui a revisão manual
# (que não escala conforme o número de notebooks cresce) por um gate
# automático.
#
# Escopo: só notebooks/**/*.py no formato "Databricks notebook source"
# (é o formato que versionamos neste repo — .ipynb não é o artefato
# deployado).
# ==============================================================================

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    (re.compile(r"^\s*%uv\s+sync\b"), "%uv sync"),
    (re.compile(r"^\s*%pip\s+install\b"), "%pip install"),
    (re.compile(r"^\s*%conda\b"), "%conda"),
]

TARGET_DIR = "notebooks"


def check_file(path: Path) -> list:
    violations = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # comentário não conta — e não deveria "proteger" nada
        for pattern, label in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                violations.append(f"{path}:{lineno}: '{label}' fora de comentário")
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / TARGET_DIR
    if not base.exists():
        return 0

    all_violations = []
    for path in base.rglob("*.py"):
        if path.is_file():
            all_violations.extend(check_file(path))

    if all_violations:
        print("Commit bloqueado — comando de ambiente dev encontrado em notebook versionado:")
        for v in all_violations:
            print(f"  {v}")
        print(
            "\nEsses comandos não podem estar no código versionado, nem comentados "
            "explicando que 'só roda em dev' — comentário não impede execução. "
            "Rode manualmente numa célula solta durante o desenvolvimento, sem "
            "commitar essa célula."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
