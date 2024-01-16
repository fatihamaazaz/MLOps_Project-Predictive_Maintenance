import joblib
import pandas as pd


def predict(data, model):
    # Chargement des encodeurs depuis les fichiers
    label_encoder = joblib.load('artifacts/encoder/encoder.pkl')
    normalizer = joblib.load('artifacts/normalizer/normalizer.pkl')

    # Transformation de la colonne 'Type' avec l'encodeur
    data['Type'] = label_encoder.transform(data['Type'])

    # Normalisation des données avec le normaliseur
    data_normalized = normalizer.transform(data)
    data = pd.DataFrame(data_normalized, columns=[column for column in data.columns])

    # Séparation des caractéristiques numériques et catégorielles
    numerical_data = data.loc[:, 'Air temperature [K]':'Tool wear [min]']
    categorical_data = data['Type']

    # Prédictions avec le modèle
    predictions = model.predict([numerical_data, categorical_data])
    predictions = (predictions > 0.5).astype(int)

    # Interprétation des prédictions
    if predictions == 1:
        message = "Risque de défaillance de la machine"
    else:
        message = "Aucun risque détecté"

    return {
        "message": message,
        "prediction": int(predictions[0][0])
    }
