from sklearn.model_selection import StratifiedKFold
from Classifiers.constants import * 
from Classifiers.utils.common import read_yaml
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score


def train_cross_validation():
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    config = b.training_evaluating
    dataset_url = config.source_data_path  # Chemin vers les données
    model_path = config.model_path  # Chemin vers le modèle de base

    # Chemin vers le fichier de paramètres
    path2 = PARAMS_FILE_PATH
    p = read_yaml(path2)  # Lecture du fichier de paramètres YAML
    params = p.NN  # Paramètres de modéle

    # Chargement du modèle pré-entraîné
    model = tf.keras.models.load_model(model_path)

    # Chargement des données depuis le fichier CSV
    data = pd.read_csv(dataset_url)

    # Séparation des caractéristiques numériques et catégorielles
    numerical_data = data.loc[:, 'Air temperature [K]':'Tool wear [min]']
    categorical_data = data['Type']

    # Étiquettes (labels)
    labels = data['Machine failure']

    # Définition de la stratégie de validation croisée (StratifiedKFold pour la classification binaire)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_list = []  # Liste pour stocker les précisions
    
    # Boucle sur les plis de la validation croisée
    for fold, (train_idx, test_idx) in enumerate(skf.split(numerical_data, labels)):
        train_numerical_data, test_numerical_data = numerical_data.iloc[train_idx], numerical_data.iloc[test_idx]
        train_categorical_data, test_categorical_data = categorical_data.iloc[train_idx], categorical_data.iloc[test_idx]
        train_labels, test_labels = labels.iloc[train_idx], labels.iloc[test_idx]

        # Boucle d'entraînement
        model.fit([train_numerical_data, train_categorical_data], train_labels, epochs=params.epochs, batch_size=params.batch_size,
                    validation_data=([test_numerical_data, test_categorical_data], test_labels))

        # Enregistrement des métriques de validation
        val_loss = model.evaluate([test_numerical_data, test_categorical_data], test_labels)

        # Après l'entraînement, évaluation sur l'ensemble de test
        predictions = model.predict([test_numerical_data, test_categorical_data])

        # Conversion des prédictions en classes (0 ou 1)
        binary_predictions = (predictions > 0.5).astype(int)

        # Calcul de la précision et enregistrement dans la liste
        accuracy = accuracy_score(test_labels, binary_predictions)
        accuracy_list.append(accuracy)

    # Calcul de la précision moyenne sur tous les plis
    mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"test_accuracy", mean_accuracy)

    return


if __name__ == '__main__':
    train_cross_validation()
