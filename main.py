from src.train import train
import logging
import os
import wandb
from src.options import NetworkOptions
from src.download_script import download_data, download_model
from src.validate_model import validate
from src.trainer import infer_model

def main():
    # setup the logger
    # 1: Download data
    # 2: Train
    # initialize wandb
    wandb.init(project="glaucoma", entity="sudonuma")
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s, %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(log_dir, "logs.log")
    )

    logger = logging.getLogger()
    
    # parse options
    options = NetworkOptions()
    opts = options.parse()

    # Download data
    download_data(logger)

    # Train model
    train(opts, logger)

    # Download model
    user_input = input(f"Do you want to download model a different model (yes/no)? ").strip().lower()
    if user_input == "yes":
        download_model(logger)

    # Evaluate model
    validate(opts, logger)

    # Infer model
    infer_model(opts, logger)


    

if __name__ == "__main__":
    main()  