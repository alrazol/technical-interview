import pytest
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data import DatasetLoader


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory for tests."""
    return tmp_path


@pytest.fixture
def dummy_parquet_files(tmp_cache_dir: Path):
    """Create fake cached parquet files."""
    df_features = pl.DataFrame({"feature_1": [1.0, 2.0], "episode_id": [0, 1]})
    df_targets = pl.DataFrame({"target": [0.5, 0.9], "episode_id": [0, 1]})
    df_features.write_parquet(tmp_cache_dir / "df_features.parquet")
    df_targets.write_parquet(tmp_cache_dir / "df_targets.parquet")
    return df_features, df_targets


class TestDatasetLoaderCaching:
    def test_loads_from_cache_when_files_exist(
        self,
        tmp_cache_dir: Path,
        dummy_parquet_files,
    ):
        _, _ = dummy_parquet_files
        with patch("grid2op.make") as mock_make:
            loader = DatasetLoader(
                env_name="test_env",
                backend=MagicMock(),
                n_busbar=2,
                cache_dir=str(tmp_cache_dir),
                episode_count=1,
                n_actions=1,
                overwrite_cache=False,
            )

            _, _ = loader.load()
            mock_make.assert_not_called()

    def test_loads_from_env_when_no_data_in_cache(self, tmp_cache_dir: Path):
        with patch("grid2op.make") as mock_make:
            loader = DatasetLoader(
                env_name="test_env",
                backend=MagicMock(),
                n_busbar=2,
                cache_dir=str(tmp_cache_dir),
                episode_count=1,
                n_actions=1,
                overwrite_cache=False,
            )
            with (
                patch.object(
                    loader,
                    "_create_realistic_observation",
                    return_value=[[MagicMock()]],
                ),
                patch.object(
                    loader,
                    "_create_training_data",
                    return_value=(pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [2]})),
                ),
            ):
                _, _ = loader.load()
            mock_make.assert_called_once()
