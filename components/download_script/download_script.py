import argparse
import logging
import os
import shutil
import zipfile

from google.cloud import storage

# TODO change download script to one script
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def download(bucket_name, source_blob_name, logger):
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), source_blob_name
    )
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(file_path)

    logger.info("Unzipping " + source_blob_name)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    logger.info("deleting " + source_blob_name)
    os.remove(file_path)


def setup(bucket_name, source_blob_name, logger):
    directory_name = os.path.splitext(source_blob_name)[0]
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

    download(bucket_name, source_blob_name, logger)


# download data or model
def go(args):
    setup(args.bucket_name, args.source_blob_name, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a dataset to you local directory.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--bucket_name",
        type=str,
        help="The name of the bucket that includes the dataset.",
        required=True,
    )

    parser.add_argument(
        "--source_blob_name",
        type=str,
        help="The name of the object to download.",
        required=True,
    )

    args = parser.parse_args()

    go(args)
