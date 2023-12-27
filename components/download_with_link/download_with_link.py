import argparse
import logging
import os
import shutil
import zipfile

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def download(url, file_name, logger):
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), file_name
    )
    logger.info("Downloading " + file_name)
    r = requests.get(url)
    open(file_path, "wb").write(r.content)
    logger.info("Unzipping " + file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    logger.info("deleting " + file_path)
    os.remove(file_path)


def setup(url, directory_name, logger):
    dir_name = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), directory_name
    )
    path = dir_name
    if os.path.exists(path):
        if directory_name == "model":
            user_input = (
                input(
                    "Are you sure you want to erase your current saved model? (yes/no)"
                )
                .strip()
                .lower()
            )
        else:
            user_input = (
                input(
                    f"Local directory {directory_name} already exists. Do you want to erase it and download the {directory_name} (yes/no)? "
                )
                .strip()
                .lower()
            )

        if user_input == "yes":
            logger.info("deleting directory")
            shutil.rmtree(path)
        else:
            logger.info("Download cancelled by the user.")
            return

    file_name = directory_name + ".zip"
    download(url, file_name, logger)


def go(args):
    setup(args.url, args.dir_name, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a dataset or trained model to you local directory with a link.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--url",
        type=str,
        help="Url of the data or the model to download.",
        required=True,
    )

    parser.add_argument(
        "--dir_name",
        type=str,
        help="The name of the object to download.",
        required=True,
    )

    args = parser.parse_args()

    go(args)
