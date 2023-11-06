import glob
import os

import pandas as pd


def test_csv_content(test_options):
    df = pd.read_csv(test_options.data_csv_path)

    assert "challenge_id" in df.columns

    for image_path in df["challenge_id"]:
        matching_files = glob.glob(
            f"./data/dataset/**/{image_path}.jpg", recursive=True
        )
        assert all(os.path.exists(file) for file in matching_files)
