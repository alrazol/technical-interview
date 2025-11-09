import typer
from pathlib import Path
from src.settings import Settings
from src.registry.on_disk_registry import OnDiskRegistry
from src.splitters.single import SingleTrainTestSplit
import src.models as models
from lightsim2grid import LightSimBackend
from src.config_loader import ExperimentConfigLoader
from src.data import DatasetLoader
from src.train import train

app = typer.Typer()
settings = Settings()

EPISODES_IN_SIM = 10
N_ACTIONS_IN_SIM = 50


@app.command()
def train_experiment(
    experiment_name: str = typer.Option(
        ..., help="Name of the experiment, must match an experiment config name."
    ),
    experiment_config_path: Path = typer.Option(
        ...,
        help="Path to the experiment config directory.",
    ),
    overide_cache: bool = typer.Option(
        False,
        help="Whether to override the dataset cache or not.",
    ),
):
    config = ExperimentConfigLoader.from_yaml(config_path=experiment_config_path)
    model = getattr(models, config.algorithm_type)(**config.hyper_parameters)
    train(
        artifacts_registry=OnDiskRegistry(
            experiment_name=experiment_name, artifacts_dir=settings.artifacts_dir
        ),
        splitter=SingleTrainTestSplit(test_size=0.2, seed=42),
        dataset_loader=DatasetLoader(
            env_name="l2rpn_case14_sandbox",
            backend=LightSimBackend(),
            n_busbar=3,
            cache_dir=settings.cache_dir,
            episode_count=EPISODES_IN_SIM,
            n_actions=N_ACTIONS_IN_SIM,
            overwrite_cache=overide_cache,
        ),
        model=model,
    )


if __name__ == "__main__":
    app()
