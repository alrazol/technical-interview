import polars as pl
from plotly import graph_objects as go
from pathlib import Path
from src.registry.base import BaseExperimentRegistry


class OnDiskRegistry(BaseExperimentRegistry):
    """
    Registry that saves experiments artifacts results on disk.
    """

    def __init__(
        self,
        experiment_name: str,
        artifacts_dir: Path,
    ) -> None:
        self.artifacts_dir = artifacts_dir / experiment_name
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def log_fig_as_artifact(self, fig: go.Figure, fig_name: str) -> None:
        with open(self.artifacts_dir / f"{fig_name}.html", "w") as file:
            fig.write_html(file)

    def log_str_as_artifact(self, content: str, file_name: str) -> None:
        with open(self.artifacts_dir / f"{file_name}.txt", "w") as file:
            file.write(content)

    def log_preds_actuals_as_parquet(
        self,
        y_preds: pl.DataFrame,
        y_true: pl.DataFrame,
        file_name: str,
    ) -> None:
        df = pl.concat([y_preds, y_true], how="horizontal")
        df.write_parquet(self.artifacts_dir / f"{file_name}.parquet")
