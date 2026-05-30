from pathlib import Path
import yaml
from pydantic import BaseModel


class Split(BaseModel):
    train: float
    valid: float
    test: float


class Cost(BaseModel):
    fp: float
    k: int


class Paths(BaseModel):
    raw_csv: str
    parquet: str
    processed: str
    models: str


class Config(BaseModel):
    seed: int
    target: str
    paths: Paths
    split: Split
    cost: Cost


def load_config(path: str | Path = "config/config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(**data)
