from src.models.base import BaseAlgorithm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from typing import Self
import polars as pl
import numpy as np


class SequentialRegressor(BaseAlgorithm):
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.classifier_model = DecisionTreeClassifier()
        self.linear_model = LinearRegression(fit_intercept=self.fit_intercept)

    def fit(self, X: pl.DataFrame, y: pl.Series) -> Self:
        y_class: pl.Series = (~y.is_finite()).cast(pl.Int8)
        finite_mask = y.is_finite()

        X_reg = X.filter(finite_mask)
        y_reg = y.filter(finite_mask)

        X_array = X.to_numpy()
        y_class_array = y_class.to_numpy().ravel()

        X_reg_array = X_reg.to_numpy()
        y_reg_array = y_reg.to_numpy().ravel()

        self.classifier_model.fit(X_array, y_class_array)
        self.linear_model.fit(X_reg_array, y_reg_array)

        return self

    def predict(self, X: pl.DataFrame) -> pl.DataFrame:
        class_preds = self.classifier_model.predict(X.to_numpy())
        reg_preds = self.linear_model.predict(X.to_numpy())
        preds = np.where(class_preds == 1, np.inf, reg_preds)
        return pl.DataFrame(preds)
