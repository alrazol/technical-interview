import polars as pl
import plotly.graph_objects as go
import plotly.express as px


def plot_errors_distribution(
    y_preds: pl.DataFrame,
    y_true: pl.DataFrame,
    nbins: int = 100,
) -> go.Figure:
    errors = y_preds - y_true
    return px.histogram(errors, nbins=nbins)
