"""Testes de src/config/data_dictionary.py (sem cluster: Spark é simulado)."""
import pytest

from src.config import data_dictionary as dd


class _FakeDF:
    def __init__(self, columns):
        self.columns = columns


class _FakeCatalog:
    def __init__(self, tables):
        self._tables = tables

    def tableExists(self, fqn):
        return fqn in self._tables


class FakeSpark:
    """Simula só o que apply_dictionary usa; registra os SQLs executados."""

    def __init__(self, tables):
        self._tables = tables
        self.catalog = _FakeCatalog(tables)
        self.statements = []

    def table(self, fqn):
        return _FakeDF(self._tables[fqn])

    def sql(self, stmt):
        self.statements.append(stmt)


def _table(name):
    return next(t for t in dd.TABLES if t.name == name)


def test_todas_as_colunas_tem_descricao():
    for t in dd.TABLES:
        for c in t.columns:
            assert c.description.strip(), f"{t.name}.{c.name} sem descrição"


def test_sem_colunas_duplicadas_por_tabela():
    for t in dd.TABLES:
        names = [c.name for c in t.columns]
        assert len(names) == len(set(names)), f"duplicada em {t.name}"


def test_gold_herda_descricao_do_silver():
    fct = {c.name: c for c in _table("fct_credit_profile").columns}
    silver = {c.name: c for c in dd.SILVER_COLUMNS}
    assert (fct["revolving_utilization"].description
            == silver["revolving_utilization_unsecured"].description)
    assert fct["target_dlq_2yrs"].source == "Silver: target_default_2yrs"


def test_markdown_contem_todas_as_colunas_e_tabelas():
    md = dd.render_markdown()
    for t in dd.TABLES:
        assert f"{t.schema}.{t.name}" in md
        for c in t.columns:
            assert f"`{c.name}`" in md


def test_markdown_versionado_esta_em_dia():
    assert dd.main(["--check"]) == 0


def test_apply_aplica_comentarios_e_tags():
    fqn = "credito_dev.gold.dim_customer"
    spark = FakeSpark({fqn: ["customer_id", "age", "num_dependents", "_gold_processed_at"]})
    report = dd.apply_dictionary(spark, "credito_dev", _table("dim_customer"))
    assert report["undocumented"] == [] and report["missing_in_table"] == []
    assert any(s.startswith(f"COMMENT ON TABLE {fqn} IS") for s in spark.statements)
    assert sum("ALTER COLUMN" in s for s in spark.statements) == 4
    assert any("SET TAGS" in s and "'camada' = 'gold'" in s for s in spark.statements)


def test_apply_ignora_tabela_inexistente():
    spark = FakeSpark({})
    report = dd.apply_dictionary(spark, "credito_dev", _table("dim_customer"))
    assert report["skipped"] is True
    assert spark.statements == []


def test_apply_reporta_coluna_sem_descricao_e_strict_falha():
    fqn = "credito_dev.gold.dim_customer"
    cols = ["customer_id", "age", "num_dependents", "_gold_processed_at", "coluna_nova"]
    report = dd.apply_dictionary(FakeSpark({fqn: cols}), "credito_dev", _table("dim_customer"))
    assert report["undocumented"] == ["coluna_nova"]
    with pytest.raises(RuntimeError):
        dd.apply_dictionary(FakeSpark({fqn: cols}), "credito_dev",
                            _table("dim_customer"), strict=True)


def test_alvo_ausente_no_scoring_nao_e_erro():
    fqn = "credito_dev.silver.give_me_some_credit_scoring"
    cols = [c.name for c in dd.SILVER_COLUMNS if c.name != "target_default_2yrs"]
    report = dd.apply_dictionary(FakeSpark({fqn: cols}), "credito_dev",
                                 _table("give_me_some_credit_scoring"), strict=True)
    assert report["missing_in_table"] == []


def test_escape_de_aspas_e_barra():
    assert dd._q("it's") == "'it\\'s'"
    assert dd._q("a\\b") == "'a\\\\b'"
