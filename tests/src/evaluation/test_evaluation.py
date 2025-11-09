import pytest
import polars as pl
import numpy as np
from src.evaluation.evaluation import Evaluation


@pytest.fixture
def simple_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Simple deterministic dataset for testing metrics."""
    y_true = pl.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
    y_preds = pl.DataFrame({"target": [1.1, 2.0, 2.9, np.inf]})
    return y_preds, y_true


@pytest.fixture
def all_inf_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Dataset where all values are inf -> should return NaN metrics."""
    y_true = pl.DataFrame({"target": [np.inf, np.inf]})
    y_preds = pl.DataFrame({"target": [np.inf, np.inf]})
    return y_preds, y_true


@pytest.fixture
def eval_instance(simple_data) -> Evaluation:
    """Fixture returning an Evaluation instance for reusable access."""
    y_preds, y_true = simple_data
    return Evaluation(y_preds=y_preds, y_true=y_true)


class TestEvaluation:
    def test_rmse_excludes_inf(self, eval_instance: Evaluation):
        rmse = eval_instance.rmse
        # Manually compute expected RMSE excluding last row with inf
        expected_rmse = np.sqrt(
            np.mean([(1.1 - 1.0) ** 2, (2.0 - 2.0) ** 2, (2.9 - 3.0) ** 2])
        )
        assert np.isclose(rmse, expected_rmse)

    def test_bias_excludes_inf(self, eval_instance: Evaluation):
        bias = eval_instance.bias
        # Exclude last row (inf)
        expected_bias = np.mean([1.1 - 1.0, 2.0 - 2.0, 2.9 - 3.0])
        assert np.isclose(bias, expected_bias)

    def test_accuracy_detects_inf(self, eval_instance: Evaluation):
        accuracy = eval_instance.accuracy
        # Only last row (inf in both preds/true?) counts as 1 mismatch here
        y_preds, y_true = eval_instance.y_preds, eval_instance.y_true
        y_preds_masked = (~np.isfinite(y_preds.to_numpy())).astype(int)
        y_true_masked = (~np.isfinite(y_true.to_numpy())).astype(int)
        expected_accuracy = np.mean(y_preds_masked == y_true_masked)
        assert np.isclose(accuracy, expected_accuracy)

    def test_repr_contains_metrics(self, eval_instance: Evaluation):
        text = repr(eval_instance)
        assert "Evaluation Metrics:" in text
        assert "Accurary:" in text
        assert "RMSE:" in text
        assert "BIAS:" in text

    def test_all_inf_returns_nan(self, all_inf_data):
        y_preds, y_true = all_inf_data
        eval_inf = Evaluation(y_preds, y_true)
        assert np.isnan(eval_inf.rmse)
        assert np.isnan(eval_inf.bias)
        # accuracy should be 1.0 because both have same inf mask
        assert eval_inf.accuracy == 1.0
