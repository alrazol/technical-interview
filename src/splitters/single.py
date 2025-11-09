import polars as pl
import numpy as np

from src.splitters.base import BaseSplitter


class SingleTrainTestSplit(BaseSplitter):
    def __init__(self, test_size=0.2, seed=None):
        self.test_size = test_size
        self.seed = seed

    def split(
        self, df: pl.DataFrame, ids: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        unique_ids = ids.unique(maintain_order=True).to_series().to_list()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(unique_ids)

        n_ids = len(unique_ids)
        n_test_ids = int(n_ids * self.test_size)

        test_ids = set(unique_ids[:n_test_ids])
        id_col = ids.columns[0]
        df_test = df.filter(pl.col(id_col).is_in(test_ids))
        df_train = df.filter(~pl.col(id_col).is_in(test_ids))

        return df_train, df_test

    def get_params(self) -> dict:
        return {"test_size": self.test_size, "seed": self.seed}
