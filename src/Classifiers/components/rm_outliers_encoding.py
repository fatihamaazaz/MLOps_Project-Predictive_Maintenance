from sklearn.preprocessing import LabelEncoder
import pandas as pd
from Classifiers.constants import *  
from Classifiers.utils.common import read_yaml 
import os
import joblib 


def outliers_encode():
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    data_config = b.data_processing
    dataset_url = data_config.source_data_path  # Chemin vers les données
    encoding_model_path = data_config.encoding_model_path  # Chemin où sauvegarder le modèle
    os.makedirs(encoding_model_path, exist_ok=True)  # Création du répertoire si inexistant
    download_dir = encoding_model_path + '/encoder.pkl'  # Chemin du modèle encodé
    data = pd.read_csv(dataset_url)  # Chargement des données depuis le fichier CSV

    col = 'Rotational speed [rpm]'

    # Calcul des quartiles pour identifier les valeurs aberrantes
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    # Définition des seuils pour les valeurs aberrantes
    seuil_inf = Q1 - 1.5 * IQR
    seuil_sup = Q3 + 1.5 * IQR

    # Suppression des valeurs aberrantes du DataFrame
    data = data[(data[col] >= seuil_inf) & (data[col] <= seuil_sup)]

    # Extraction des colonnes pertinentes du DataFrame
    df = data.loc[:, "Type":"Machine failure"]

    # Label Encoding pour la colonne 'Type'
    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])

    # Sauvegarde du modèle d'encodage
    joblib.dump(label_encoder, download_dir)

    # Retourne le DataFrame modifié
    return df
