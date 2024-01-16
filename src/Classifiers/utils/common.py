import yaml
from box import ConfigBox 


def read_yaml(path_to_yaml):
    # Ouvre le fichier YAML spécifié par path_to_yaml
    with open(path_to_yaml) as fichier_yaml:
        # Charge le contenu YAML en utilisant PyYAML
        contenu = yaml.safe_load(fichier_yaml)  
        # encapsuler le contenu et le renvoyer
        return ConfigBox(contenu)