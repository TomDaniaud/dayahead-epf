"""Walk-forward (rolling-origin) backtesting with configurable retraining policy.

Timeline for each test day D:
  1. forecast D with the current model (trained on days < last retrain date);
  2. the day-ahead auction for D clears -> realised prices of D are known, so the daily
     error is observed *before* forecasting D+1 and fed to the drift monitor;
  3. if the policy says so, retrain on the window ending at D (inclusive) for D+1.
No information from days > D ever reaches the model forecasting D+1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from epf.data import TARGET
from epf.drift import PerformanceDriftMonitor, psi
from epf.features import daily_matrix
from epf.metrics import forecast_kpis
from epf.models import Forecaster, NaiveWeekly


@dataclass
class BacktestConfig:
    test_start: str
    test_end: str | None = None
    retrain: Literal["never", "periodic", "drift", "periodic+drift"] = "periodic+drift"
    retrain_every: int = 30  # days, for periodic policies
    min_days_between_retrains: int = 7
    window: Literal["expanding", "rolling"] = "rolling"
    window_days: int = 730
    drift_delta: float = 0.1
    drift_threshold: float = 4.0
    drift_ref_days: int = 14


@dataclass
class BacktestResult:
    model: str
    forecasts: pd.DataFrame  # (n_days, 24)
    actuals: pd.DataFrame
    naive: pd.DataFrame
    daily_mae: pd.Series
    retrains: list[dict] = field(default_factory=list)
    fit_seconds: float = 0.0

    @property
    def kpis(self) -> dict:
        k = forecast_kpis(self.actuals, self.forecasts, self.naive)
        k["n_retrains"] = len(self.retrains)
        k["fit_seconds"] = round(self.fit_seconds, 1)
        return k

    def to_long(self) -> pd.DataFrame:
        f = self.forecasts.stack().rename("forecast")
        a = self.actuals.stack().rename("actual")
        out = pd.concat([a, f], axis=1)
        out.index = [d + pd.Timedelta(hours=int(h)) for d, h in out.index]
        out.index.name = "timestamp"
        return out


def walk_forward(model: Forecaster, df: pd.DataFrame, cfg: BacktestConfig,
                 verbose: bool = False) -> BacktestResult:
    P = daily_matrix(df[TARGET])
    days = P.index
    end = pd.Timestamp(cfg.test_end) if cfg.test_end else days[-1]
    test_days = days[(days >= pd.Timestamp(cfg.test_start)) & (days <= end)]
    if len(test_days) == 0:
        raise ValueError("empty test period")

    model.prepare(df)
    naive = NaiveWeekly()
    naive.prepare(df)
    monitor = PerformanceDriftMonitor(cfg.drift_ref_days, cfg.drift_delta, cfg.drift_threshold)
    use_drift = "drift" in cfg.retrain
    use_periodic = "periodic" in cfg.retrain

    def window(first_forecast_day: pd.Timestamp) -> pd.DatetimeIndex:
        w = days[days < first_forecast_day]
        if cfg.window == "rolling":
            w = w[w >= first_forecast_day - pd.Timedelta(days=cfg.window_days)]
        return w

    fit_time, retrains = 0.0, []

    def refit(day, reason, stat=None):
        nonlocal fit_time
        train = window(day)
        t0 = time.perf_counter()
        model.fit(train)
        fit_time += time.perf_counter() - t0
        res = (df["load_fc"] - df["wind_fc"] - df["solar_fc"])
        recent = res[(res.index >= day - pd.Timedelta(days=30)) & (res.index < day)]
        retrains.append({"day": day, "reason": reason, "train_days": len(train),
                         "ph_stat": stat,
                         "psi_residual_load": psi(res[res.index.normalize().isin(train)].values,
                                                  recent.values)})
        monitor.reset()
        if verbose:
            print(f"[{model.name}] retrain {day.date()} ({reason}), {len(train)} days")

    refit(test_days[0], "initial")
    since, preds, nv, errs = 0, {}, {}, {}
    for i, d in enumerate(test_days):
        preds[d] = model.predict(d)
        nv[d] = naive.predict(d)
        errs[d] = float(np.nanmean(np.abs(preds[d] - P.loc[d].to_numpy(dtype=float))))
        since += 1
        if i == len(test_days) - 1:
            break
        nxt = test_days[i + 1]
        drift_alarm = use_drift and monitor.update(errs[d])
        if use_periodic and since >= cfg.retrain_every:
            refit(nxt, "periodic")
            since = 0
        elif drift_alarm and since >= cfg.min_days_between_retrains:
            refit(nxt, "drift", monitor.ph.statistic)
            since = 0

    idx = pd.DatetimeIndex(list(preds))
    mk = lambda dct: pd.DataFrame(np.vstack(list(dct.values())), index=idx, columns=range(24))
    return BacktestResult(model.name, mk(preds), P.loc[idx], mk(nv),
                          pd.Series(errs, name="daily_MAE"), retrains, fit_time)
