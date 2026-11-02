"""dayahead-epf: day-ahead electricity price forecasting toolkit."""

from epf.backtest import BacktestConfig, BacktestResult, walk_forward
from epf.metrics import diebold_mariano, forecast_kpis

__all__ = ["BacktestConfig", "BacktestResult", "diebold_mariano", "forecast_kpis", "walk_forward"]
__version__ = "0.1.0"
