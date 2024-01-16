from sklearn.preprocessing import QuantileTransformer
from Classifiers.constants import *  
from Classifiers.utils.common import read_yaml
import os
import pandas as pd

def features_distribution(data):
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    data_config = b.data_processing
    download_dir = data_config.local_data_file + '/processed_data.csv'  # Chemin pour sauvegarder les données traitées
    os.makedirs(data_config.local_data_file, exist_ok=True)  # Création du répertoire si inexistant

    # Colonnes à transformer
    columns = ['Air temperature [K]', 'Process temperature [K]',
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Transformation de la distribution des colonnes
    for column in columns:
        transformer = QuantileTransformer(output_distribution='normal')
        # Remplacement de la colonne existante par les valeurs transformées
        data[column] = transformer.fit_transform(data[[column]])

    # Sauvegarde des données traitées au format CSV
    data.to_csv(download_dir, index=False)

    return
