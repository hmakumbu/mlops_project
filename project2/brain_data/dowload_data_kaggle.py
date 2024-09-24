import os
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

def download_dataset(dataset, download_path):
    """Download a Kaggle dataset to a specified path.

    Args:
        dataset (str): The Kaggle dataset identifier (e.g., 'awsaf49/brats20-dataset-training-validation').
        download_path (str): The directory path to download the dataset to.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print(f"Dataset {dataset} downloaded to {download_path}")

if __name__ == "__main__":
    load_dotenv()

    dataset = os.getenv('DATASET_NAME')
    download_path = os.getenv('DATASET_PATH')

    download_dataset(dataset, download_path)

  