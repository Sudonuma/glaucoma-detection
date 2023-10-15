# Glaucoma Challenge

Welcome to the Glaucoma Challenge repository. This repository contains the code for the Glaucoma Challenge project.

![Ohne Titel](https://user-images.githubusercontent.com/2522480/149497318-fe47c02c-696a-4cb5-8841-2dbe6785029d.png)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [data](#data)
- [CSV](#CSV)
- [TODO](#TODO)

## Prerequisites

Having Conda installed is the recommended approach. If not, you can install the requirements directly, although it's not the preferred method.

## Usage

1. clone the directory.

```bash
git clone https://github.com/Sudonuma/MLCodingChallenge.git
```

2. cd to MLCodingChallenge.

```bash
cd MLCodingChallenge
```

3. Create a conda environment and activate it. (if you have conda installed otherwise you can directly install the requirements)

```bash
conda create --name glaucomaenv python=3.9
```

```bash
conda activate glaucomaenv
```

4. Install the dependencies.

```bash
pip install -r requirements.txt
```

5. run the main script for training, validation and inference (see [file](src/options.py) for available arguments, you can also run with default args).

```bash
python main.py
```

6. When asked about wandb API key please copy and paste the token I included in the email. Note that the Token will be deleted in 2/3 days, if you need more time please do not hesitate to contact me.


> Note : If you would like to only validate and infer on a model you can run this command `python main.py --validate_only True`. it will ask you to download a pretrained model if you want.
## Data

The glaucodma dataset contains images categorized into folders labeled from 0 to 5. These folders include photographs of patients, some of whom have glaucoma, while others do not. Notably, the number of images of individuals without glaucoma significantly surpasses those with the condition.

The dataset is accompanied by a train_labels.csv file, which maps each image's name to its corresponding label. To simplify the labels, we encoded them as follows: "rg" represents class 1, and "ngr" corresponds to class 0. The resulting dataset with these encoded labels is saved in a file called encoded_dataset.csv.

For our experimentation, we set aside a portion (10%) of the data as a test dataset, and the information for these test samples is stored in a CSV file called `test_data.csv`.

In a separate experiment, we aimed to mitigate class imbalance by reducing the dataset size. We generated two distinct files: `reduced_encoded_train_data.csv` for training and `reduced_encoded_test_data.csv` for testing.

Additionally, to facilitate code testing for other users, we included two CSV files, namely, `dummy_train_data.csv` and `dummy_test_data.csv`. These files enable users to test the code without the need to download the entire 50+GB dataset.

**Should you wish to train on the full dataset, you can do so by copying the image folders (0 to 5) into the `data/dataset` directory. Make sure to adjust the `--data_csv_path` argument to point to `./data/dataset/encoded_train_dataset.csv` and the `--test_data_csv_path` to `./data/dataset/encoded_test_dataset.csv`.**

If you want to train on the balanced data, you can utilize the `reduced_encoded_train_data.csv` and `reduced_encoded_test_data.csv` files.

The data is split 70% for training, 20% validation and 10% for model evaluation (testing).

## CSV

1. To train the model on all the dataset use: `train_data.csv` and `test_data.csv`
2. To train the model on balanced data (exactly the sample number of samples for each class): `reduced_encoded_train_data.csv` and `reduced_encoded_test_data.csv`
3. To train the model on Downsampled but not well balanced data use: `13ktrain_data.csv` and `13ktest_data.csv`
4. `dummy_train_data.csv` and `dummy_test_data.csv` are just for the purpose to run the code.

## TODO

1. csv files should be tracked with DVC.
2. Improve the EDA and the pre-processing step.
3. Test the output of the model.
4. Add more tests.
5. Optimise the stratfied sampling.
6. Use Siamese Neural Networks as it is robust against unbalanced.
7. Lint.
