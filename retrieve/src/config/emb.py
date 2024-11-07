import pydantic
import yaml

from .base import EnvYaml

class TextEncoderYaml(pydantic.BaseModel):
    name: str

class EmbYaml(pydantic.BaseModel):
    env: EnvYaml
    entity_identifier_file: str
    text_encoder: TextEncoderYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'emb'
    
    return EmbYaml(**yaml_data).model_dump()
