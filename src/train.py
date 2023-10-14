import wandb
from options import NetworkOptions
from trainer import Trainer


options = NetworkOptions()
opts = options.parse()

if __name__ == "__main__":
    
    wandb.init(project="glaucoma", entity="sudonuma")
    trainer = Trainer(opts)
    trainer.train()
    