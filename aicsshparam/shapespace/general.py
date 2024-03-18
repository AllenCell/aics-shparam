import yaml
from pathlib import Path


def load_config_file(path: Path, fname="config.yaml"):
    with open(Path(path)/fname, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
