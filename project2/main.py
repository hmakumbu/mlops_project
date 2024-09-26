
from project2.app.eda import DataGenerator
from project2.app.load_data import Datasource
from project2.app.model import Unet

import os

from dotenv import load_dotenv

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




#  est Of the model 
if __name__ == "__main__":
    
    dataset = os.getenv('DATASET_NAME')
    download_path = os.getenv('DATASET_PATH')
    
    load_dotenv()
    source = Datasource()
    source.download_dataset(dataset, download_path)
    
    
    # __________________________________________Data Spliting_____________________________________#
    # Split the Dataset
    train_and_test_ids = source.pathListIntoIds()
    train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2)
    train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)
    
    
    
    
    #________________________________________DataGenerator__________________________________#
    
    training_generator = DataGenerator(train_ids)
    validation_generator = DataGenerator(val_ids)
    test_generator = DataGenerator(test_ids)

    
    #______________________________________Initialisation de la classe Unet_________________#
    unet = Unet(img_size=128, num_classes=4)

    # Compilation du modèle
    unet.compile_model()

    # Affichage du modèle sous forme d'image
    unet.plot_model('unet_model.png')

    # Entraînement du modèle
    history = unet.train(training_generator, validation_generator, epochs=35, train_ids=train_ids)

    # Sauvegarde du modèle entraîné
    unet.save_model('my_model.keras')

    # Chargement du modèle
    unet.load_model('my_model.keras')

















