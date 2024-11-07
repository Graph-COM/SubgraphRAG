import pydantic
import yaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)
