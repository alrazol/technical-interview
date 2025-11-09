from src.registry.base import BaseExperimentRegistry
from src.data import DatasetLoader
from src.splitters.base import BaseSplitter
from src.models.base import BaseAlgorithm
from src.evaluation.evaluation import Evaluation
from src.plotting import plot_errors_distribution
import polars as pl
from src.logger import logger


def train(
    artifacts_registry: BaseExperimentRegistry,
    splitter: BaseSplitter,
    dataset_loader: DatasetLoader,
    model: BaseAlgorithm,
) -> None:
    logger.info(
        "Loading dataset",
        overide_cache=dataset_loader.overwrite_cache,
        cache_dir=str(dataset_loader.cache_dir),
    )
    X, y = dataset_loader.load()
    logger.info(
        "Splitting dataset",
        splitter=splitter.__class__.__name__,
        splitter_args=splitter.get_params(),
    )
    ids = X.select(pl.col("episode_id"))
    y = y.drop("episode_id")
    features_cols = X.drop("episode_id").columns
    target_cols = y.columns
    df = pl.concat([X, y], how="horizontal")
    df_train, df_test = splitter.split(df, ids=ids)
    X_train, X_test, y_train, y_test = (
        df_train.select(features_cols),
        df_test.select(features_cols),
        df_train.select(target_cols),
        df_test.select(target_cols),
    )
    logger.info(
        "Training model",
        model=model.__class__.__name__,
        X_train_shape=X_train.shape,
        y_train_shape=y_train.shape,
    )
    model.fit(X_train, y_train.to_series())
    y_preds = model.predict(X_test)
    logger.info(
        "Evaluating model", model=model.__class__.__name__, X_test_shape=X_test.shape
    )
    y_preds_insample = model.predict(X_train)
    evaluator_test, evaluator_train = (
        Evaluation(y_preds=y_preds, y_true=y_test),
        Evaluation(y_preds=y_preds_insample, y_true=y_train),
    )
    artifacts_registry.log_str_as_artifact(
        str(evaluator_test), file_name="metrics_test"
    )
    artifacts_registry.log_str_as_artifact(
        str(evaluator_train), file_name="metrics_train"
    )
    artifacts_registry.log_fig_as_artifact(
        plot_errors_distribution(y_preds=y_preds, y_true=y_test, nbins=1000),
        "errors_distribution_test",
    )
    artifacts_registry.log_fig_as_artifact(
        plot_errors_distribution(y_preds=y_preds_insample, y_true=y_test, nbins=1000),
        "errors_distribution_train",
    )
    artifacts_registry.log_preds_actuals_as_parquet(
        y_preds=pl.DataFrame({"y_pred": y_preds}),
        y_true=y_test,
        file_name="preds_vs_actuals_test",
    )
    artifacts_registry.log_preds_actuals_as_parquet(
        y_preds=pl.DataFrame({"y_pred": y_preds_insample}),
        y_true=y_train,
        file_name="preds_vs_actuals_train",
    )
    logger.info(
        "Test metrics",
        mape=round(float(evaluator_test.mape), 3),
        rmse=round(float(evaluator_test.rmse), 3),
        bias=round(float(evaluator_test.bias), 3),
    )
    logger.info("Training complete")
