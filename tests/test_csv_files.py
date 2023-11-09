import os

import pandas as pd
import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Data and model testing will be included later in CI."
)
def test_train_csv_not_empty(test_options):
    df = pd.read_csv(test_options.data_csv_path)
    assert len(df) > 2


def test_test_csv_not_empty(test_options):
    df = pd.read_csv(test_options.test_data_csv_path)
    assert len(df) > 2
