from imblearn.over_sampling import SMOTE

def balancing(data):
    # Séparation des features et des labels
    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']

    # Utilisation de SMOTE pour suréchantillonner les données et équilibrer les classes
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Reconstruction du DataFrame équilibré avec les labels
    X_resampled['Machine failure'] = y_resampled

    # Retourne le DataFrame équilibré
    return X_resampled
