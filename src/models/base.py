from abc import ABC, abstractmethod
import polars as pl
from typing import Self


class BaseAlgorithm(ABC):
    @abstractmethod
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | pl.Series, **kwargs) -> Self: ...

    @abstractmethod
    def predict(self, X: pl.DataFrame, **kwargs) -> pl.DataFrame: ...
