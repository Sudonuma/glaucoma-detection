import logging
import os
import shutil
import zipfile

from google.cloud import storage


def download(bucket_name, source_blob_name, logger):
    file_path = "./" + source_blob_name
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(source_blob_name)

    logger.info("Unzipping " + source_blob_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(".")
    logger.info("deleting " + source_blob_name)
    os.remove(file_path)


def setup(bucket_name, source_blob_name, logger):
    directory_name = os.path.splitext(source_blob_name)[0]
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

    download(bucket_name, source_blob_name, logger)


def download_data(bucket_name, logger):
    setup(bucket_name, "data.zip", logger)


def download_model(bucket_name, logger):
    setup(bucket_name, "model.zip", logger)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    download_model("glaucoma-dataset", logger)
