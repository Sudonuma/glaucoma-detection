import glob
import os

import pandas as pd
import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Data and model testing will be included later in CI."
)
def test_csv_content(test_options):
    df = pd.read_csv(test_options.data_csv_path)

    assert "challenge_id" in df.columns

    for image_path in df["challenge_id"]:
        matching_files = glob.glob(
            f"./data/dataset/**/{image_path}.jpg", recursive=True
        )
        assert all(os.path.exists(file) for file in matching_files)
