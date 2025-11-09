import pytest
import polars as pl
import numpy as np

from src.splitters.single import SingleTrainTestSplit


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a simple dataframe with a group id column."""
    n = 100
    group_ids = np.repeat(np.arange(10), 10)  # 10 groups of 10
    return pl.DataFrame({"id": group_ids, "x": np.arange(n), "y": np.random.randn(n)})


@pytest.fixture
def ids_dataframe(sample_dataframe: pl.DataFrame) -> pl.DataFrame:
    """Return a dataframe of IDs corresponding to the id column."""
    return sample_dataframe.select("id")


class TestSingleTrainTestSplit:
    def test_split_sizes(self, sample_dataframe: pl.DataFrame, ids_dataframe: pl.DataFrame):
        splitter = SingleTrainTestSplit(test_size=0.3)
        df_train, df_test = splitter.split(sample_dataframe, ids_dataframe)

        unique_ids = ids_dataframe.unique().to_series().to_list()
        expected_test_ids = int(len(unique_ids) * 0.3)
        expected_train_ids = len(unique_ids) - expected_test_ids

        # Count how many unique ids are in each split
        train_ids = set(df_train["id"].unique().to_list())
        test_ids = set(df_test["id"].unique().to_list())

        assert len(train_ids) == expected_train_ids
        assert len(test_ids) == expected_test_ids
        assert train_ids.isdisjoint(test_ids)

    def test_split_disjoint(self, sample_dataframe: pl.DataFrame, ids_dataframe: pl.DataFrame):
        splitter = SingleTrainTestSplit(test_size=0.2)
        df_train, df_test = splitter.split(sample_dataframe, ids_dataframe)

        train_ids = set(df_train["id"].to_list())
        test_ids = set(df_test["id"].to_list())

        assert train_ids.isdisjoint(test_ids), "Train and test sets should have disjoint ids"

    def test_deterministic_split_with_seed(self, sample_dataframe: pl.DataFrame, ids_dataframe: pl.DataFrame):
        splitter_1 = SingleTrainTestSplit(test_size=0.3, seed=42)
        splitter_2 = SingleTrainTestSplit(test_size=0.3, seed=42)

        train_1, test_1 = splitter_1.split(sample_dataframe, ids_dataframe)
        train_2, test_2 = splitter_2.split(sample_dataframe, ids_dataframe)

        # Ensure deterministic selection of ids
        assert set(train_1["id"].to_list()) == set(train_2["id"].to_list())
        assert set(test_1["id"].to_list()) == set(test_2["id"].to_list())

    def test_split_with_zero_test_size(self, sample_dataframe: pl.DataFrame, ids_dataframe: pl.DataFrame):
        splitter = SingleTrainTestSplit(test_size=0.0)
        df_train, df_test = splitter.split(sample_dataframe, ids_dataframe)

        assert df_test.shape[0] == 0
        assert df_train.shape[0] == sample_dataframe.shape[0]

    def test_split_with_full_test_size(self, sample_dataframe: pl.DataFrame, ids_dataframe: pl.DataFrame):
        splitter = SingleTrainTestSplit(test_size=1.0)
        df_train, df_test = splitter.split(sample_dataframe, ids_dataframe)

        assert df_train.shape[0] == 0
        assert df_test.shape[0] == sample_dataframe.shape[0]

    def test_get_params(self):
        splitter = SingleTrainTestSplit(test_size=0.4, seed=123)
        params = splitter.get_params()
        assert params == {"test_size": 0.4, "seed": 123}
