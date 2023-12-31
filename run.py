import logging
import os

import wandb
from src.download_script import download_data, download_model
from src.options import NetworkOptions
from src.train_model.train import train
from src.train_model.trainer import infer_model
from src.validate_model import validate


def main():
    # initialize wandb
    wandb.init(project="glaucoma", entity="sudonuma")

    # setup the logger
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_formatter = logging.Formatter(
        "%(asctime)s, %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup File handler
    file_handler = logging.FileHandler("./logs/logs.log")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Setup Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)

    # Get our logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add both Handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # parse options
    options = NetworkOptions()
    opts = options.parse()

    if opts.validate_only:
        # Download or not model
        logger.info("Validation and Inference only mode:")
        user_input = (
            input(
                "Do you want to download a different model. (your current model will be DELETED) (yes/no)? "
            )
            .strip()
            .lower()
        )
        if user_input == "yes":
            download_model(opts.model_url, logger)

        # Evaluate model
        validate(opts, logger)

        # Infer model
        infer_model(opts, logger)
        return

    # Download data
    download_data(opts.data_url, logger)

    # Train model
    train(opts, logger)

    # Evaluate model
    validate(opts, logger)

    # Infer model
    infer_model(opts, logger)


if __name__ == "__main__":
    main()
