import os
import shutil
import zipfile

import requests


def download(url, file_name, logger):
    file_path = "./" + file_name
    logger.info("Downloading " + file_name)
    r = requests.get(url)
    open(file_name, "wb").write(r.content)
    logger.info("Unzipping " + file_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(".")
    logger.info("deleting " + file_name)
    os.remove(file_path)


def setup(url, directory_name, logger):
    path = "./" + directory_name
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


def download_data(data_url, logger):
    setup(data_url, "data", logger)


def download_model(model_url, logger):
    setup(model_url, "model", logger)
