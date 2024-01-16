# Projet de Maintenance Prédictive (AI4I 2020 Dataset) - MLOps

Ce projet est développé dans le cadre du cours de "Méthodologie pour la science des données" pour appliquer les concepts de MLOps à la maintenance prédictive.

## Objectif

Le principal objectif de ce projet est de mettre en place un pipeline MLOps pour automatiser le flux de travail associé à la maintenance prédictive.

## Dataset

Dans ce projet, nous utilisons une base de données synthétique issue du jeu de données AI4I 2020 Predictive Maintenance. Cette base de données reflète des scénarios réalistes liés à la maintenance prédictive.

## Outils

- Contrôle de version avec Git
- Gestion de version des données avec DVC
- Suivi des expériences avec MLflow
- Déploiement de modèle avec une API Flask

## Installation de projet

### Etape 01

Cloner le répertoire

```bash
https://github.com/fatihamaazaz/MLOps_Project.git
```

Ou télécharger le projet et initialiser git

```bash
git init
```

### Etape 02

Créer un environnement conda et l'activer après avoir ouvert le dossier du projet

```bash
conda create -n env_name python=3.11 -y
```

```bash
conda activate env_name
```

### Etape 03

Installer les requirements

```bash
pip install -r requirements.txt
```

### Etape 04

Initialiser dvc

```bash
dvc init
```

### Etape 04

Exécuter le pipeline

```bash
dvc repro
```

### Etape 05

Suivre la performance des modèles

```bash
mlflow ui
```

### Etape 06

Déployer la solution

```bash
python app.py
```
