import polars as pl
from abc import ABC, abstractmethod
import plotly.graph_objects as go


class BaseExperimentRegistry(ABC):
    """
    Base class for developping experiments artifacts registries.
    """

    @abstractmethod
    def log_fig_as_artifact(self, fig: go.Figure, fig_name: str) -> None: ...

    @abstractmethod
    def log_str_as_artifact(self, content: str, file_name: str) -> None: ...

    @abstractmethod
    def log_preds_actuals_as_parquet(
        self,
        y_preds: pl.DataFrame,
        y_true: pl.DataFrame,
        file_name: str,
    ) -> None: ...
