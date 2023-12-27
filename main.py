import os

import hydra
import mlflow
from omegaconf import DictConfig

import wandb


# This automatically reads in the configuration
@hydra.main(config_name="config")
def go(config: DictConfig):
    wandb.init(project="glaucoma-detection", entity="sudonuma")
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

    # _ = mlflow.run(
    #     os.path.join(root_path, "components/download_with_link"),
    #     "main",
    #     parameters={
    #         "url": config["data"]["url"],
    #         "dir_name": config["data"]["dir_name"],
    #     },
    # )
    _ = mlflow.run(
        os.path.join(root_path, "src/train_model"),
        "main",
        parameters={
            "data_path": config["trainer"]["data_path"],
            "data_csv_path": config["trainer"]["data_csv_path"],
            "test_data_csv_path": config["trainer"]["test_data_csv_path"],
            "model_path": config["trainer"]["model_path"],
            "image_path": config["trainer"]["image_path"],
            "data_url": config["trainer"]["data_url"],
            "batch_size": config["trainer"]["batch_size"],
            "lr": config["trainer"]["lr"],
            "num_epochs": config["trainer"]["num_epochs"],
            "validate_only": config["trainer"]["validate_only"],
        },
    )


if __name__ == "__main__":
    go()
