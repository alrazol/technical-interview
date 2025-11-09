import polars as pl
import numpy as np

ROUNDING = 2


class Evaluation:
    def __init__(self, y_preds: pl.DataFrame, y_true: pl.DataFrame) -> None:
        self.y_preds = y_preds
        self.y_true = y_true

    @property
    def accuracy(self) -> float:
        return self.compute_accurary(self.y_preds, self.y_true)

    @property
    def mae(self) -> float:
        return self.compute_mae(self.y_preds, self.y_true)

    @property
    def rmse(self) -> float:
        return self.compute_rmse(self.y_preds, self.y_true)

    @property
    def bias(self) -> float:
        return self.compute_bias(self.y_preds, self.y_true)

    @property
    def mape(self) -> float:
        return self.compute_mape(self.y_preds, self.y_true)

    @staticmethod
    def compute_rmse(y_preds: pl.DataFrame, y_true: pl.DataFrame) -> float:
        y_true_np = y_true.to_numpy()
        y_preds_np = y_preds.to_numpy()
        mask = np.isfinite(y_true_np) & np.isfinite(y_preds_np)
        if not np.any(mask):
            return float("nan")
        y_true_clean = y_true_np[mask]
        y_preds_clean = y_preds_np[mask]
        squared_diff = (y_true_clean - y_preds_clean) ** 2
        return np.sqrt(squared_diff.mean())

    @staticmethod
    def compute_mape(y_preds: pl.DataFrame, y_true: pl.DataFrame) -> float:
        y_true_np = y_true.to_numpy()
        y_preds_np = y_preds.to_numpy()
        mask = (
            np.isfinite(y_true_np)
            & np.isfinite(y_preds_np)
            & (np.abs(y_true_np) > 1e-8)
        )
        if not np.any(mask):
            return float("nan")
        perc_err = np.abs((y_true_np[mask] - y_preds_np[mask]) / y_true_np[mask])
        return 100 * perc_err.mean()

    @staticmethod
    def compute_mae(y_preds: pl.DataFrame, y_true: pl.DataFrame) -> float:
        y_true_np = y_true.to_numpy()
        y_preds_np = y_preds.to_numpy()
        mask = np.isfinite(y_true_np) & np.isfinite(y_preds_np)
        if not np.any(mask):
            return float("nan")
        diff = np.abs(y_true_np[mask] - y_preds_np[mask])
        return diff.mean()

    @staticmethod
    def compute_bias(y_preds: pl.DataFrame, y_true: pl.DataFrame) -> float:
        y_true_np = y_true.to_numpy()
        y_preds_np = y_preds.to_numpy()
        mask = np.isfinite(y_true_np) & np.isfinite(y_preds_np)
        if not np.any(mask):
            return float("nan")
        y_true_clean = y_true_np[mask]
        y_preds_clean = y_preds_np[mask]
        bias = y_preds_clean - y_true_clean
        return bias.mean()

    @staticmethod
    def compute_accurary(y_preds: pl.DataFrame, y_true: pl.DataFrame) -> float:
        y_preds_array, y_true_array = y_preds.to_numpy(), y_true.to_numpy()
        y_preds_masked, y_true_masked = (
            (~np.isfinite(y_preds_array)).astype(int),
            (~np.isfinite(y_true_array)).astype(int),
        )
        return np.mean(y_preds_masked == y_true_masked)

    def __repr__(self):
        template = [
            "Evaluation Metrics:",
            f"Accurary:             {round(self.accuracy, ROUNDING)}",
            f"MAE:                 {round(self.mae, ROUNDING)}",
            f"MAPE:                {round(self.mape, ROUNDING)}%",
            f"RMSE:                 {round(self.rmse, ROUNDING)}",
            f"BIAS:                 {round(self.bias, ROUNDING)}",
        ]
        return "\n".join(template)
