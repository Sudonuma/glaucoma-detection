import os

import pytest

from src.train_model.trainer import evaluate_model

# sys.path.append(os.path.abspath(os.path.join(os.path.pardir, "data")))


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Data and model testing will be included later in CI."
)
def test_evaluate_model(test_options, logger):

    accuracy, precision, recall, f1, auc = evaluate_model(test_options, logger)

    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert isinstance(auc, float)
