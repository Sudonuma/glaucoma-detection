import os
import requests
import zipfile
import shutil
import sys


def download_data(logger):
     url = "https://glaucoma-dataset-009.s3.eu-central-1.amazonaws.com/data.zip"
     logger.info('Downloading data.zip')
     r = requests.get(url)
     open('data.zip' , 'wb').write(r.content)
     logger.info('Unzipping data')
     with zipfile.ZipFile('./data.zip', 'r') as zip_ref:
          zip_ref.extractall('.')
     logger.info('removing data.zip')
     os.remove("./data.zip")



def setup_data(logger):
     if os.path.exists('./data'):
          user_input = input(f"Local directory data already exists. Do you want to erase it and download? the data (yes/no)? ").strip().lower()
          logger.info(f"deleting data folder")
          if user_input == "yes":
               shutil.rmtree('./data')

          else:
               logger.info("Download cancelled by the user.")
               sys.exit()

     download_data(logger)

