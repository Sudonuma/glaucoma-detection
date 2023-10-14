import torchvision.datasets as datasets
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
from torch import optim
import os
from time import sleep
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from multiprocessing import Process, Manager
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import logging
from data.screenings import ResnetDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve


logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(message)s",datefmt='%Y-%m-%d %H:%M:%S', filename="./logs/logs.log")
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def not_stratified_train_val_dataset(dataset, val_split=0.15):

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def train_val_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    
    import numpy as np
    # labels = [dataset[i][1] for i in range(len(dataset))]
    print('start stratifying 0')
    data_array = np.array(dataset)

    # extract the labels
    labels = data_array[:, 1]
    print('start stratifying 1')
    

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    print('start stratifying 2')
    train_idx, temp_idx = next(splitter.split(labels, labels))
    print('ending datasplit')
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, temp_idx)

    datasets = {
        'train': train_subset,
        'val': val_subset
    }
    
    return datasets

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.model = models.resnet50(pretrained=True) # change this to device
        self.n_epochs = self.opt.num_epochs
        self.batch_size=self.opt.batch_size
        self.lr = self.opt.lr
        self.data_path = self.opt.data_path
        self.data_csv_path = self.opt.data_csv_path
        self.best_val_loss = float('inf')
        self.best_model_state = None


        self.transformation = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

        
        self.test_transformation = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            ])


        self.dataset = ResnetDataset(root_dir=self.data_path, csv_file=self.data_csv_path, transform=self.transformation)
    
        # TODO Perform stratified train-validation-test split
        
        
        data_split = not_stratified_train_val_dataset(self.dataset)


        self.train_dataloader = DataLoader(data_split['train'],
                                shuffle=True,
                                num_workers=8,
                                batch_size=self.batch_size)

        self.val_dataloader = DataLoader(data_split['val'],
                                shuffle=True,
                                num_workers=2,
                                batch_size=self.batch_size)


        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9)
        num_features = self.model.fc.in_features
        num_classes = 2  
        self.model.fc = nn.Linear(num_features, num_classes)

        wandb.watch(self.model, log='all')        
        self.model= self.model.to(device)

    def train(self):
        logger.info(f"Started training")
        for epoch in range(self.n_epochs):
            self.run_epoch(epoch)
            self.validate(epoch)
            # Save the model at the best epoch
            # if self.best_model_state is not None:
            #     self.save_model("./model", self.best_model_state, epoch)
        logger.info(f"Finished training")
        logger.info(f"Exporting model")
        # self.save_model("./model", self.model, epoch)

    def run_epoch(self, epoch):
    
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
                sleep(0.1) # To see te preogressive bar
                # TODO log the average running loss not the loss per item
                wandb.log({'Train Loss': loss.item()})


    def validate(self, epoch):
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
        average_running_loss = sum(cumulative_loss)/(len(cumulative_loss))
        print(f"validation loss {average_running_loss}\n")
        wandb.log({'Validation Loss': average_running_loss})

        # Check if the current validation loss is better than the best
        if average_running_loss < self.best_val_loss:
            self.best_val_loss = average_running_loss
            # Save the model state at the best epoch
            self.best_model_state = self.model.state_dict()
            self.save_model("./model", self.best_model_state, epoch)


    def save_model(self, save_path, model, epoch):
            """Save model weights of best epoch to disk
            """
            # save_folder = os.path.join(save_path, "models", f"{model_name}.pth")
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)
            save_path = os.path.join(save_path, f"modelWeights.pth")
            torch.save(model, save_path)
            # TODO save model as artifact in wandb


def load_model(model, model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()


def evaluate_model(options):

    data_path = options.data_path
    test_data_csv_path = options.test_data_csv_path
    model_path = options.model_path
    
    transformation =  transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
    dataset = ResnetDataset(root_dir=data_path, csv_file=test_data_csv_path, transform=transformation)
    test_dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=1)

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
    wandb.log({"AUC score": auc, "F1 Score": f1, "Accuracy": accuracy, "Precision": precision, "Recall": recall})
    
    # log confusion matrix
    wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_predictions)})

    # Log the ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
    roc_curve_data = wandb.Table(data=list(zip(fpr, tpr, thresholds)), columns=["fpr", "tpr", "thresholds"])

    # Log the ROC curve table
    wandb.log({"ROC Curve": wandb.plot.line(roc_curve_data, "fpr", "tpr", title="ROC Curve")})

    
    return accuracy, precision, recall, f1, auc


def infer_model(options):
    # Load the saved model
    from PIL import Image
    

    image_path = options.image_path
    model_path = options.model_path
    
    transformation =  transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

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

    return predicted_class, class_probabilities

# for testing
if __name__ == "__main__":    
    from options import NetworkOptions

    options = NetworkOptions()
    opts = options.parse()
    predicted_class, class_probabilities = infer_model(opts)
    print(predicted_class, class_probabilities)