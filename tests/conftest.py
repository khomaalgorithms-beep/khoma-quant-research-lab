
import numpy as np
import pandas as pd
import pytest

from tests.helpers import prices_from_returns


@pytest.fixture
def random_prices() -> pd.Series:
    """Reproducible random-walk prices (seeded)."""
    rng = np.random.default_rng(7)
    return prices_from_returns(rng.normal(0.0004, 0.01, 600))
