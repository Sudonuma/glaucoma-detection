import wandb
from options import NetworkOptions
from trainer import Trainer

def main() -> None:
    """
    Main function for training a glaucoma detection model.

    Initializes Weights and Biases (wandb), sets project and entity, and initiates training.

    Returns:
        None
    """
    
    options = NetworkOptions()
    opts = options.parse()

    wandb.init(project="glaucoma", entity="sudonuma")
    trainer = Trainer(opts)
    trainer.train()

if __name__ == "__main__":
    main()