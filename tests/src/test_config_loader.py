import yaml
import pytest
from pathlib import Path
from src.config_loader import ExperimentConfigLoader


@pytest.fixture
def example_yaml_config():
    return {
        "comment": "This is a test experiment.",
        "model": {
            "algorithm": "SequentialRegressor",
            "hyper_parameters": {
                "fit_intercept": True,
            },
        },
    }


def test_experiment_config_loader_from_yaml(tmp_path: Path, example_yaml_config):
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(example_yaml_config, f)
    config = ExperimentConfigLoader.from_yaml(yaml_path)
    assert config.comment == "This is a test experiment."
    assert config.algorithm_type == "SequentialRegressor"
    assert config.hyper_parameters["fit_intercept"] is True
