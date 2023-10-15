from src.trainer import Trainer

def train(options, logger) -> None:
    """
    Main function for training a glaucoma detection model.

    Initializes Weights and Biases (wandb), sets project and entity, and initiates training.

    Returns:
        None
    """

    
    trainer = Trainer(options, logger)
    trainer.train()

if __name__ == "__main__":
    train()
