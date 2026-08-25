import pytest


@pytest.fixture
def sample_dataframe():
    """Fixture compartilhada -- substituir por amostra real do caso."""
    import pandas as pd
    return pd.DataFrame({"col_exemplo": [1, 2, 3]})
