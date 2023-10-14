from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import glob


def pil_loader(path: str) -> Image:
    """
    Load an image from the specified path.

    Args:
        path (str): The file path to the image.

    Returns:
        Image: A PIL Image in RGB format.
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ResnetDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transform=None):
        """
        Initialize a custom dataset for ResNet.

        Args:
            root_dir (str): The root directory where the image files are located.
            csv_file (str): The path to a CSV file containing image file names and labels.
            transform (callable, optional): A function/transform to apply to the images.

        Returns:
            None
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        img_name = self.df.iloc[idx, 0]
        file_path = glob.glob(f'{self.root_dir}/**/{img_name}.jpg', recursive=True)
        
        image = pil_loader(file_path[0])
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label
