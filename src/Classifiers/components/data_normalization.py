from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from Classifiers.constants import *  
from Classifiers.utils.common import read_yaml 
import os

def normalize(data):
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    data_config = b.data_processing
    normalizing_model_path = data_config.normalizing_model_path  # Chemin où sauvegarder le modèle de normalisation
    os.makedirs(normalizing_model_path, exist_ok=True)  # Création du répertoire si inexistant
    download_dir = normalizing_model_path + '/normalizer.pkl'  # Chemin du modèle de normalisation
    df = data.loc[:, "Type":"Tool wear [min]"]  # Sélection des colonnes pertinentes du DataFrame

    # Normalisation avec StandardScaler
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df)

    # Sauvegarde du modèle de normalisation
    joblib.dump(scaler, download_dir)

    # Création d'un nouveau DataFrame avec les données normalisées
    df_normalized = pd.DataFrame(data_normalized, columns=[column for column in df.columns])

    # Ajout de la colonne 'Machine failure' au DataFrame normalisé
    df_normalized['Machine failure'] = data['Machine failure']

    return df_normalized
