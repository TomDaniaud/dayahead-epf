"""Forecast performance KPIs used in the EPF literature (Lago et al., 2021)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def forecast_kpis(y: pd.DataFrame, yhat: pd.DataFrame, naive: pd.DataFrame | None = None) -> dict:
    """KPIs on (n_days, 24) matrices.

    MAE / RMSE in EUR/MWh; sMAPE (%) is reported but unstable near zero prices, which is
    why rMAE (MAE relative to the naive benchmark, < 1 means better than naive) is the
    headline scale-free metric. ``bias`` > 0 means systematic over-forecasting.
    """
    e = (yhat - y).to_numpy(dtype=float)
    a, f = y.to_numpy(dtype=float), yhat.to_numpy(dtype=float)
    out = {
        "MAE": np.nanmean(np.abs(e)),
        "RMSE": np.sqrt(np.nanmean(e ** 2)),
        "sMAPE": 100 * np.nanmean(2 * np.abs(e) / (np.abs(a) + np.abs(f) + 1e-9)),
        "bias": np.nanmean(e),
        "MAE_p95": np.nanpercentile(np.abs(e), 95),  # tail error, matters for trading risk
    }
    if naive is not None:
        out["rMAE"] = out["MAE"] / np.nanmean(np.abs((naive - y).to_numpy(dtype=float)))
    return {k: float(v) for k, v in out.items()}


def mae_by_hour(y: pd.DataFrame, yhat: pd.DataFrame) -> pd.Series:
    return (yhat - y).abs().mean(axis=0).rename("MAE")


def diebold_mariano(y: pd.DataFrame, f1: pd.DataFrame, f2: pd.DataFrame, norm: int = 1) -> dict:
    """Multivariate Diebold-Mariano test (Ziel & Weron, 2018), one-sided.

    Daily loss differential d_t = ||e1_t||_norm - ||e2_t||_norm over the 24-h vector, so
    serial correlation across hours of the same day does not inflate the statistic.
    H0: forecast 2 is not more accurate than forecast 1. Small p-value -> f2 significantly
    better than f1.
    """
    e1 = np.abs((f1 - y).to_numpy(dtype=float)) ** norm
    e2 = np.abs((f2 - y).to_numpy(dtype=float)) ** norm
    d = np.nanmean(e1, axis=1) - np.nanmean(e2, axis=1)
    d = d[~np.isnan(d)]
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"DM": float(stat), "p_value": float(1 - stats.norm.cdf(stat))}
