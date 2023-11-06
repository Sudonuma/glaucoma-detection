from src.trainer import evaluate_model


def validate(options, logger) -> None:
    """
    Initialize Weights and Biases (wandb), parse network options, and evaluate the model's performance.
    Print the evaluation metrics (accuracy, precision, recall, f1).
    """
    logger.info("Evaluate model")
    accuracy, precision, recall, f1, auc = evaluate_model(options, logger)
