import wandb
from options import NetworkOptions
from trainer import evaluate_model

options = NetworkOptions()
opts = options.parse()

if __name__ == "__main__":    

    wandb.init(project="glaucoma", entity="sudonuma")
    options = NetworkOptions()
    opts = options.parse()
    accuracy, precision, recall, f1, auc = evaluate_model(opts)
    print(accuracy, precision, recall, f1)