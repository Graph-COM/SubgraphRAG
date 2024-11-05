import pydantic
import yaml

class EnvYaml(pydantic.BaseModel):
    num_threads: int
    seed: int

class TextEncoderYaml(pydantic.BaseModel):
    name: str

class EmbYaml(pydantic.BaseModel):
    env: EnvYaml
    text_encoder: TextEncoderYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'emb'
    
    return EmbYaml(**yaml_data).model_dump()
