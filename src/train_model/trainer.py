import os
import sys
from time import sleep
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

import wandb

# TODO change the path to glaucoma=detectioninstead of src
# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the root directory to the Python path if not already included
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# if root_dir not in sys.path:
sys.path.append(root_dir)


from data.screenings import ResnetDataset  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# change this with logger
print(device)

wandb.init(project="glaucoma-detection", entity="sudonuma")


def not_stratified_train_val_dataset(
    dataset: Dataset, val_split: float = 0.2
) -> Dict[str, Dataset]:
    """
    Split a dataset into training and validation sets, without stratification.

    Args:
        dataset (Dataset): The dataset to split.
        val_split (float): The fraction of data to put in the validation set.

    Returns:
        dict: A dictionary with 'train' and 'val' keys, containing training and validation subsets.
    """
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split
    )
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets


class Trainer:
    def __init__(self, options, logger):
        self.opt = options
        self.model = models.resnet50(pretrained=True)  # change this to device
        self.n_epochs = self.opt.num_epochs
        self.batch_size = self.opt.batch_size
        self.lr = self.opt.lr
        self.data_path = self.opt.data_path
        self.data_csv_path = self.opt.data_csv_path
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.logger = logger

        self.transformation = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.test_transformation = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.dataset = ResnetDataset(
            root_dir=self.data_path,
            csv_file=self.data_csv_path,
            transform=self.transformation,
        )

        # TODO Perform stratified train-validation-test split

        data_split = not_stratified_train_val_dataset(self.dataset)

        self.train_dataloader = DataLoader(
            data_split["train"], shuffle=True, num_workers=8, batch_size=self.batch_size
        )

        self.val_dataloader = DataLoader(
            data_split["val"], shuffle=True, num_workers=2, batch_size=self.batch_size
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        num_features = self.model.fc.in_features
        num_classes = 2
        self.model.fc = nn.Linear(num_features, num_classes)

        wandb.watch(self.model, log="all")
        self.model = self.model.to(device)

    def train(self):
        """
        Train the model.
        """
        self.logger.info("Started training")
        for epoch in range(self.n_epochs):
            self.run_epoch(epoch)
            self.validate(epoch)

        self.logger.info("Finished training")
        self.logger.info("Exporting model")

    def run_epoch(self, epoch: int):
        """
        Run one training epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        with tqdm(self.train_dataloader, unit="batch") as tepoch:
            for i, (inputs, labels) in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                image = wandb.Image(inputs[0])

                wandb.log({"Train images example:": image})
                # wandb.log({"Train images example label:": labels[0]})

                # pass the inputs to the network
                outputs = self.model(inputs)

                # Compute loss function
                loss = self.loss_fn(outputs, labels)

                loss.backward()
                self.optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)  # To see te preogressive bar
                # TODO log the average running loss not the loss per item
                wandb.log({"Train Loss": loss.item()})

    def validate(self, epoch):
        """
        Perform model validation.

        Args:
            epoch (int): The current epoch.
        """
        cumulative_loss = []
        for i, (inputs, labels) in enumerate(self.val_dataloader, 0):
            self.model.eval()
            inputs, labels = inputs.to(device), labels.to(device)

            self.optimizer.zero_grad()

            # Pass the inputs to the network
            outputs = self.model(inputs)

            # Compute loss function
            loss = self.loss_fn(outputs, labels)

            cumulative_loss.append(loss.item())
            # wandb.log({'Validation Loss': loss.item()})
        average_running_loss = sum(cumulative_loss) / (len(cumulative_loss))
        print(f"validation loss {average_running_loss}\n")
        wandb.log({"Validation Loss": average_running_loss})

        # Check if the current validation loss is better than the best
        if average_running_loss < self.best_val_loss:
            self.best_val_loss = average_running_loss
            # Save the model state at the best epoch
            self.best_model_state = self.model.state_dict()
            self.save_model(self.best_model_state, epoch)

    def save_model(self, model, epoch):
        """Save model weights of best epoch to disk"""
        model_folder = os.path.join(os.getcwd(), "model")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model_filename = os.path.join(model_folder, "modelWeights.pth")
        torch.save(model, model_filename)
        # TODO save model as artifact in wandb


def load_model(model: nn.Module, model_path: str) -> None:
    """
    Load model weights and set the model to evaluation.

    Args:
        model (nn.Module): The PyTorch model to load weights into.
        model_path (str): The path to the model weights.
    """
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()


def evaluate_model(options, logger) -> Tuple[float, float, float, float, float]:
    """
    Evaluate a model's performance on a test dataset.

    Args:
        options: Network options.

    Returns:
        Tuple[float, float, float, float, float]:
        A tuple containing accuracy, precision, recall, F1 score, and AUC score.
    """
    # the path for dataset
    data_path = options.data_path
    test_data_csv_path = options.test_data_csv_path
    model_path = options.model_path

    transformation = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ResnetDataset(
        root_dir=data_path, csv_file=test_data_csv_path, transform=transformation
    )
    test_dataloader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=1)

    model = models.resnet50()

    num_features = model.fc.in_features
    num_classes = 2
    model.fc = nn.Linear(num_features, num_classes)

    load_model(model, model_path)
    # TODO log the test images
    all_labels = []
    all_predictions = []  # predicted labels
    all_probabilities = []  # Probabilities of the predicted labels

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.cpu().numpy())

            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probabilities.extend(probabilities)

    # compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probabilities)

    # log metrics to wandb
    wandb.log(
        {
            "AUC score": auc,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
        }
    )
    logger.info(
        f"AUC score: {auc}, F1 Score: {f1}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}"
    )
    # log confusion matrix
    wandb.log(
        {
            "Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=all_labels, preds=all_predictions
            )
        }
    )

    # ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
    roc_curve_data = wandb.Table(
        data=list(zip(fpr, tpr, thresholds)), columns=["fpr", "tpr", "thresholds"]
    )

    # Log the ROC curve table
    wandb.log(
        {"ROC Curve": wandb.plot.line(roc_curve_data, "fpr", "tpr", title="ROC Curve")}
    )

    return accuracy, precision, recall, f1, auc


def infer_model(options, logger) -> Tuple[int, List[float]]:
    """
    Perform inference on an image using a pre-trained model.

    Args:
        options (NetworkOptions): An object containing various network options.
        image_path (str): The path to the input image for inference.

    Returns:
        Tuple[int, List[float]]: A tuple containing the predicted class and class probabilities.
    """

    image_path = options.image_path
    model_path = options.model_path

    transformation = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess the input image
    image = Image.open(image_path)
    image = transformation(image)
    image = image.unsqueeze(0)

    # Load the model
    model = models.resnet50()

    num_features = model.fc.in_features
    num_classes = 2
    model.fc = nn.Linear(num_features, num_classes)

    load_model(model, model_path)

    # inference
    with torch.no_grad():
        output = model(image)

    predicted_class = output.argmax().item()

    # class probabilities
    class_probabilities = torch.softmax(output, dim=1).squeeze().tolist()
    logger.info(
        f"predicted_class {predicted_class}, class_probabilities{class_probabilities}"
    )
    return predicted_class, class_probabilities
