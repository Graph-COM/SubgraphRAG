import pydantic

class EnvYaml(pydantic.BaseModel):
    num_threads: int
    seed: int
