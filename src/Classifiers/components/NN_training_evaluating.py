from sklearn.model_selection import train_test_split
from Classifiers.constants import *  
from Classifiers.utils.common import read_yaml
import tensorflow as tf
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
import mlflow
import mlflow.keras


def train_test():
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    config = b.training_evaluating
    dataset_url = config.source_data_path  # Chemin vers les données
    model_path = config.model_path  # Chemin vers le modèle pré-entraîné

    # Chemin vers le fichier de paramètres
    path2 = PARAMS_FILE_PATH
    p = read_yaml(path2)  # Lecture du fichier de paramètres YAML
    params = p.NN  # Paramètres du modéle

    # Chargement du modèle de base
    model = tf.keras.models.load_model(model_path)

    # Chargement des données depuis le fichier CSV
    data = pd.read_csv(dataset_url)

    # Séparation des caractéristiques numériques et catégorielles
    numerical_data = data.loc[:, 'Air temperature [K]':'Tool wear [min]']
    categorical_data = data['Type']

    # Étiquettes (labels)
    labels = data['Machine failure']

    # Division des données en ensembles d'entraînement, de validation et de test
    train_numerical_data, test_numerical_data, train_categorical_data, test_categorical_data, train_labels, test_labels = train_test_split(
        numerical_data, categorical_data, labels, test_size=0.2, random_state=42)

    # Division à nouveau des données de test pour obtenir un ensemble de validation
    val_numerical_data, test_numerical_data, val_categorical_data, test_categorical_data, val_labels, test_labels = train_test_split(
        test_numerical_data, test_categorical_data, test_labels, test_size=0.5, random_state=42
    )

    # Activation de l'enregistrement automatique avec MLflow
    mlflow.tensorflow.autolog(registered_model_name="NN model")

    with mlflow.start_run() as run:

        # Boucle d'entraînement
        model.fit([train_numerical_data, train_categorical_data], train_labels, epochs=params.epochs, batch_size=params.batch_size, validation_data=([val_numerical_data, val_categorical_data], val_labels))

        # Enregistrement des métriques de validation
        val_loss = model.evaluate([val_numerical_data, val_categorical_data], val_labels)
            
        # Après l'entraînement, évaluation sur l'ensemble de test
        predictions = model.predict([test_numerical_data, test_categorical_data])

        # Conversion des prédictions en classes (0 ou 1)
        binary_predictions = (predictions > 0.5).astype(int)

        # Calcul des métriques de performance
        precision = precision_score(test_labels, binary_predictions, average='binary')
        recall = recall_score(test_labels, binary_predictions, average='binary')
        accuracy = accuracy_score(test_labels, binary_predictions)

        # Enregistrement des métriques de performance dans MLflow
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_accuracy", accuracy)

    return
