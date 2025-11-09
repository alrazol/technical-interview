import yaml
from typing import Self
from pathlib import Path
from pydantic import BaseModel, ConfigDict


class ExperimentConfigLoader(BaseModel):
    comment: str
    algorithm_type: str
    hyper_parameters: dict

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_yaml(cls, config_path: Path) -> Self:
        with open(config_path, "r") as f:
            config: dict = yaml.safe_load(f)
        return cls(
            comment=config.get("comment", ""),
            algorithm_type=config["model"]["algorithm"],
            hyper_parameters=config["model"]["hyper_parameters"],
        )
