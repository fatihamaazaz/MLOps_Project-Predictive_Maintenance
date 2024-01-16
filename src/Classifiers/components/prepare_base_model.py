# Importation des bibliothèques nécessaires
import tensorflow as tf
from tensorflow.keras import layers, models
from Classifiers.constants import *  
from Classifiers.utils.common import read_yaml 
import os


def prepare_model():
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    config = b.prepare_base_model
    download_dir = config.base_model_path + '/base_model.h5'  # Chemin pour sauvegarder le modèle
    os.makedirs(config.base_model_path, exist_ok=True)  # Création du répertoire si inexistant

    # Définition des entrées pour les caractéristiques numériques
    numerical_input = layers.Input(shape=(5,), name='numerical_input')
    numerical_dense = layers.Dense(64, activation='relu')(numerical_input)

    # Caractéristiques catégorielles (supposant qu'elles ont été encodées avec LabelEncoder)
    categorical_input = layers.Input(shape=(1,), name='categorical_input')
    categorical_dense = layers.Dense(16, activation='relu')(categorical_input)

    # Concaténation des caractéristiques numériques et catégorielles
    concatenated = layers.Concatenate()([numerical_dense, categorical_dense])

    # Couches denses pour les caractéristiques combinées
    x = layers.Dense(128, activation='relu')(concatenated)
    x = layers.Dropout(0.5)(x)  # Dropout optionnel pour la régularisation
    output = layers.Dense(1, activation='sigmoid')(x)  # Classification binaire, donc utilisation de l'activation sigmoïde

    # Création du modèle
    model = models.Model(inputs=[numerical_input, categorical_input], outputs=output)

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Affichage du résumé du modèle
    model.summary()

    # Sauvegarde du modèle
    model.save(download_dir)

    return
