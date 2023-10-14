import wandb
from options import NetworkOptions
from trainer import evaluate_model

def main() -> None:
    """
    Initialize Weights and Biases (wandb), parse network options, and evaluate the model's performance.
    Print the evaluation metrics (accuracy, precision, recall, f1).
    """
    wandb.init(project="glaucoma", entity="sudonuma")
    options = NetworkOptions()
    opts = options.parse()
    accuracy, precision, recall, f1, auc = evaluate_model(opts)
    print(accuracy, precision, recall, f1)

if __name__ == "__main__":
    main()