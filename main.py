import os

import hydra
import mlflow
from omegaconf import DictConfig


# Read hydra config
@hydra.main(config_name="config")
def go(config: DictConfig):
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # # Add the root directory to the Python path if not already included
    # print(sys.path)
    # if root_dir not in sys.path:
    #     sys.path.append(root_dir)

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

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
            "model_url": config["trainer"]["model_url"],
            "batch_size": config["trainer"]["batch_size"],
            "lr": config["trainer"]["lr"],
            "num_epochs": config["trainer"]["num_epochs"],
            "validate_only": config["trainer"]["validate_only"],
        },
    )


if __name__ == "__main__":
    go()
