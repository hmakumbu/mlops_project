from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

import os
import cv2
import random
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps
import nibabel as nib
import keras
# import keras.backend as K
import tensorflow.keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
# from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras.layers import preprocessing



class Datasource:
    global TRAIN_DATASET_PATH
    TRAIN_DATASET_PATH = os.getenv('DATASET_NAME')
    
    def __init__(self) -> None:
        pass

    def download_dataset(self, dataset, download_path):
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

    dataset = os.getenv('DATASET_NAME')
    download_path = os.getenv('DATASET_PATH')
    
    load_dotenv()
    source = Datasource()
    source.download_dataset(dataset, download_path)