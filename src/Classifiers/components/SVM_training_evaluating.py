from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
from Classifiers.constants import * 
from Classifiers.utils.common import read_yaml
import pandas as pd 


def svm_model():
    # Chemin vers le fichier de configuration
    path = CONFIG_FILE_PATH
    b = read_yaml(path)  # Lecture du fichier de configuration YAML
    config = b.training_evaluating
    dataset_url = config.source_data_path  # Chemin vers les données

    data = pd.read_csv(dataset_url)  # Chargement des données depuis le fichier CSV

    X = data.drop('Machine failure', axis=1)  # Features
    y = data['Machine failure']  # Labels

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création d'un ensemble de validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Début de l'exécution MLflow
    with mlflow.start_run() as run:
        # Initialisation du modèle SVM
        svm_model = SVC(kernel='linear', C=1.0)

        # Entraînement du modèle sur l'ensemble d'entraînement
        svm_model.fit(X_train, y_train)

        # Prédictions sur l'ensemble d'entraînement
        y_train_pred = svm_model.predict(X_train)

        # Calcul de la précision sur l'ensemble d'entraînement
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Prédictions sur l'ensemble de validation
        y_val_pred = svm_model.predict(X_val)

        # Calcul de la précision sur l'ensemble de validation
        val_accuracy = accuracy_score(y_val, y_val_pred)

        # Prédictions sur l'ensemble de test
        y_test_pred = svm_model.predict(X_test)

        # Calcul de la précision sur l'ensemble de test
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Enregistrement manuel des paramètres
        mlflow.log_params({"kernel": 'linear', "C": 1.0})

        # Enregistrement des métriques
        mlflow.log_metric("accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Sauvegarde du modèle
        mlflow.sklearn.log_model(svm_model, "svm_model")

    return
