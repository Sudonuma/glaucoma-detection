import os

import hydra
import mlflow
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name="config")
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    # TODO use this to fix saving data
    root_path = hydra.utils.get_original_cwd()

    # _ = mlflow.run(
    #     os.path.join(root_path, "components/download_script"),
    #     "main",
    #     parameters={
    #         "bucket_name": config["data"]["bucket_name"],
    #         "source_blob_name": config["data"]["source_blob_name"],
    #     },
    # )

    _ = mlflow.run(
        os.path.join(root_path, "components/download_with_link"),
        "main",
        parameters={
            "url": config["data"]["url"],
            "dir_name": config["data"]["dir_name"],
        },
    )


if __name__ == "__main__":
    go()
