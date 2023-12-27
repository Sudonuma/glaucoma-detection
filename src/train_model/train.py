import argparse
import logging
import os
import sys

from trainer import Trainer

# import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.pardir, "data")))


# sys.path.append(os.path.abspath(os.path.join('..', 'data')))

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train(args, logger) -> None:
    """
    Main function for training a glaucoma detection model.

    Initializes Weights and Biases (wandb), sets project and entity, and initiates training.

    Returns:
        None
    """

    trainer = Trainer(args, logger)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Resnet50 model.", fromfile_prefix_chars="@"
    )

    # parser.add_argument(
    #     "--artifact_name", type=str, help="Name for the artifact", required=True
    # )

    # parser.add_argument(
    #     "--artifact_type", type=str, help="Type for the artifact", required=True
    # )

    # parser.add_argument(
    #     "--artifact_description",
    #     type=str,
    #     help="Description for the artifact",
    #     required=True,
    # )

    parser.add_argument(
        "--data_path",
        type=str,
        help="path to the training data",
        default="/home/sudonuma/Documents/glaucoma-detection/data/dataset",
    )

    parser.add_argument(
        "--data_csv_path",
        type=str,
        help="path to the training data csv file.",
        default="/home/sudonuma/Documents/glaucoma-detection/data/dataset/dummy_train_data.csv",
    )

    parser.add_argument(
        "--test_data_csv_path",
        type=str,
        help="path to the test data csv file.",
        default="/home/sudonuma/Documents/glaucoma-detection/data/dataset/dummy_test_data.csv",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the saved model.",
        default="/home/sudonuma/Documents/glaucoma-detection/model/modelWeights.pth",
    )
    # label is 1
    parser.add_argument(
        "--image_path",
        type=str,
        help="path to test image (for inference).",
        default="/home/sudonuma/Documents/glaucoma-detection/data/dataset/1/TRAIN021661.jpg",
    )

    parser.add_argument(
        "--data_url",
        type=str,
        help="Dataset CS URL.",
        default="https://storage.googleapis.com/glaucoma-dataset/data.zip",
    )

    parser.add_argument(
        "--model_url",
        type=str,
        help="Model CS URL.",
        default="https://storage.googleapis.com/glaucoma-dataset/model.zip",
    )

    # label is 0
    # self.parser.add_argument("--image_path",
    #                          type=str,
    #                          help="path to the saved model.",
    #                          default="/home/sudonuma/Documents/glaucoma/data/dataset/1/TRAIN032248.jpg")

    # OPTIMIZATION options
    parser.add_argument("--batch_size", type=int, help="batch size", default=2)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=2)

    parser.add_argument(
        "--validate_only",
        type=bool,
        help="Validate and infer on existing model",
        default=False,
    )

    args = parser.parse_args()

    train(args, logger)
