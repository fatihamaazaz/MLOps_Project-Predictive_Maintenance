import flask
import mlflow
import mlflow.keras
from flask import jsonify, render_template, request
from Classifiers.components.predict import predict
from Classifiers.constants import *  
from Classifiers.utils.common import read_yaml
import pandas as pd

# Création de l'application Flask
app = flask.Flask(__name__)
app.config["debug"] = True

# Route pour la page d'accueil
@app.route("/")
def home():
    # Rendu de la page d'accueil
    return render_template("index.html")

# Route pour la prédiction (utilisant la méthode POST)
@app.route("/predict", methods=["POST"])
def predict_class():
    # Nom du modèle et sa version
    path = PARAMS_FILE_PATH
    p = read_yaml(path)  # Lecture du fichier de paramètres YAML
    params = p.prediction
    model_name = params.model
    model_version = params.version

    # Chargement du modèle depuis MLflow
    loaded_model = mlflow.keras.load_model(f"models:/{model_name}/{model_version}")

    # Récupération des données du formulaire
    type = request.form.get("type")
    air_temp = float(request.form.get("airTemperature"))
    process_temp = float(request.form.get("processTemperature"))
    rot_temp = float(request.form.get("rotation"))
    torq_temp = float(request.form.get("torque"))
    tool = float(request.form.get("tool"))

    # Création d'un DataFrame avec les données du formulaire
    data = {
        'Type': [type],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rot_temp],
        'Torque [Nm]': [torq_temp],
        'Tool wear [min]': [tool],
    }
    df = pd.DataFrame(data)

    # Prédiction avec le modèle chargé
    pred = predict(df, loaded_model)

    # Rendu de la réponse au format JSON
    return jsonify(pred)

# Point d'entrée pour exécuter l'application Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
