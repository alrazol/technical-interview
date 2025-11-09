import polars as pl
from abc import ABC, abstractmethod


class BaseSplitter(ABC):
    """
    Base class for data splitters.
    """

    @abstractmethod
    def split(
        self, df: pl.DataFrame, ids: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Splits the data into training and testing sets.

        Args:
            data: The dataset to be split.

        Returns:
            A tuple containing the training and testing sets.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Returns the parameters of the splitter.

        Returns:
            A dictionary containing the parameters of the splitter.
        """
        pass
