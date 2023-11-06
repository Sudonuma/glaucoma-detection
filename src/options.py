import argparse
import os

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class NetworkOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            "--data_path",
            type=str,
            help="path to the training data",
            default="./data/dataset",
        )

        self.parser.add_argument(
            "--data_csv_path",
            type=str,
            help="path to the training data csv file.",
            default="./data/dataset/dummy_train_data.csv",
        )

        self.parser.add_argument(
            "--test_data_csv_path",
            type=str,
            help="path to the test data csv file.",
            default="./data/dataset/dummy_test_data.csv",
        )

        self.parser.add_argument(
            "--model_path",
            type=str,
            help="path to the saved model.",
            default="./model/modelWeights.pth",
        )
        # label is 1
        self.parser.add_argument(
            "--image_path",
            type=str,
            help="path to the saved model.",
            default="./data/dataset/1/TRAIN021661.jpg",
        )

        self.parser.add_argument(
            "--data_url",
            type=str,
            help="Dataset S3 URL.",
            default="https://glaucoma-dataset-009.s3.eu-central-1.amazonaws.com/data.zip",
        )

        self.parser.add_argument(
            "--model_url",
            type=str,
            help="Model S3 URL.",
            default="https://glaucoma-dataset-009.s3.eu-central-1.amazonaws.com/model.zip",
        )

        # label is 0
        # self.parser.add_argument("--image_path",
        #                          type=str,
        #                          help="path to the saved model.",
        #                          default="/home/sudonuma/Documents/glaucoma/data/dataset/1/TRAIN032248.jpg")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=2)
        self.parser.add_argument(
            "--lr", type=float, help="learning rate", default=0.001
        )
        self.parser.add_argument(
            "--num_epochs", type=int, help="number of epochs", default=2
        )

        self.parser.add_argument(
            "--validate_only",
            type=bool,
            help="Validate and infer on existing model",
            default=False,
        )

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
