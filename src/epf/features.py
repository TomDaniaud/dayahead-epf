"""Leak-free feature construction for day-ahead forecasting.

Information set when bidding for delivery day D (gate closure D-1, 12:00):
  * prices of D-1 and earlier (D-1 prices were published at ~13:00 on D-2),
  * day-ahead forecasts of load / wind / solar for D,
  * the calendar.
Every price feature below therefore uses data from days strictly before D. The unit test
``test_no_leakage`` perturbs prices of day D and checks the features of D are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epf.data import EXOG, TARGET


def daily_matrix(series: pd.Series) -> pd.DataFrame:
    """Hourly series (24 rows per day) -> (n_days, 24) matrix indexed by day."""
    s = series.copy()
    frame = pd.DataFrame({"v": s.values, "day": s.index.normalize(), "hour": s.index.hour})
    return frame.pivot(index="day", columns="hour", values="v")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (day, hour) with hour as a feature and ``y`` as target.

    A single gradient-boosting model over all 24 hours shares statistical strength across
    hours while ``hour`` and the hour-specific lags let it specialise.
    """
    P = daily_matrix(df[TARGET])
    days = P.index

    def lag(k: int) -> pd.DataFrame:  # price of day D-k, same hour
        return P.shift(k)

    d1 = P.shift(1)
    feats = {
        "p_d1": lag(1), "p_d2": lag(2), "p_d3": lag(3), "p_d7": lag(7),
        "p_d1_mean": _bcast(d1.mean(axis=1), days),
        "p_d1_min": _bcast(d1.min(axis=1), days),
        "p_d1_max": _bcast(d1.max(axis=1), days),
        "p_d1_h23": _bcast(d1[23], days),  # most recent known price
        "p_d7_mean": _bcast(P.shift(1).rolling(7).mean().mean(axis=1), days),
    }
    for col in EXOG:
        X = daily_matrix(df[col])
        feats[col] = X  # forecast for D: allowed
        feats[f"{col}_d1"] = X.shift(1)
        feats[f"{col}_dmean"] = _bcast(X.mean(axis=1), days)
    res = daily_matrix(df["load_fc"] - df["wind_fc"] - df["solar_fc"])
    feats["residual_fc"] = res
    feats["residual_fc_dmean"] = _bcast(res.mean(axis=1), days)

    # Level normalisation: express price lags relative to the D-1 daily mean. Trees cannot
    # extrapolate outside the training range, so after a level shift (e.g. the 2021-22 gas
    # crisis) raw-price trees fail badly; relative features + a relative target fix this.
    level = feats["p_d1_mean"]
    for k in ["p_d1", "p_d2", "p_d3", "p_d7", "p_d1_min", "p_d1_max", "p_d1_h23", "p_d7_mean"]:
        feats[k] = feats[k] - level

    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long.index.names = ["day", "hour"]
    long["y"] = P.stack(future_stack=True)
    d = long.index.get_level_values("day")
    long["hour"] = long.index.get_level_values("hour")
    long["dow"] = d.dayofweek
    long["month"] = d.month
    long["doy_sin"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    long["doy_cos"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    return long


def _bcast(daily: pd.Series, days: pd.Index) -> pd.DataFrame:
    """Repeat a per-day value across the 24 hourly columns."""
    return pd.DataFrame(np.repeat(daily.reindex(days).values[:, None], 24, axis=1),
                        index=days, columns=range(24))


def feature_columns(long: pd.DataFrame) -> list[str]:
    return [c for c in long.columns if c != "y"]
