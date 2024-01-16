import os
import gdown
from Classifiers.constants import *  # Import des constantes
from Classifiers.utils.common import read_yaml  # Import de la fonction read_yaml pour lire les fichiers YAML


def download_file():
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    data_config = b.data_ingestion
    dataset_url = data_config.source_URL  # URL du fichier à télécharger
    download_dir = data_config.local_data_file + '/raw_data.csv'  # Chemin local pour sauvegarder le fichier
    os.makedirs(data_config.local_data_file, exist_ok=True)  # Création du répertoire si inexistant
    file_id = dataset_url.split("/")[-2]
    prefix = 'https://drive.google.com/uc?/export=download&id='
    
    # Téléchargement du fichier depuis Google Drive
    gdown.download(prefix + file_id, download_dir)

    return
