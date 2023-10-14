import os
import requests
import zipfile
import shutil
import sys
import logging

DATA_URL = "https://glaucoma-dataset-009.s3.eu-central-1.amazonaws.com/data.zip"
MODAL_URL = "https://glaucoma-dataset-009.s3.eu-central-1.amazonaws.com/model.zip"


logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

def download(url,file_name, logger):
     file_path = './' + file_name
     logger.info('Downloading ' + file_name)
     r = requests.get(url)
     open(file_name , 'wb').write(r.content)
     logger.info('Unzipping ' + file_name)
     with zipfile.ZipFile(file_path, 'r') as zip_ref:
          zip_ref.extractall('.')
     logger.info('deleting ' +  file_name)
     os.remove(file_path)



def setup(url, directory_name, logger):
     path = './' +  directory_name
     if os.path.exists(path):

          user_input = input(f"Local directory " + directory_name + " already exists. Do you want to erase it and download the data (yes/no)? ").strip().lower()

          if user_input == "yes":
               logger.info(f"deleting directory")
               shutil.rmtree(path)
          else:
               logger.info("Download cancelled by the user.")
               sys.exit()

     file_name = directory_name + '.zip'
     download(url, file_name, logger)


def download_data(logger): 
     setup(DATA_URL,'data',logger)


def download_model(logger): 
     setup(MODAL_URL,'model',logger)


download_model(logger)