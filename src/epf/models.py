"""Forecasters sharing one interface.

    model.prepare(df)          # once per backtest: cache features on the full dataset
    model.fit(train_days)      # train on the given delivery days only
    model.predict(day) -> (24,) array of prices for delivery day ``day``

``prepare`` sees the whole dataset for speed, but every feature for day D is built from
information available at D-1 gate closure (see ``epf.features``), and ``fit`` only ever
sees rows whose delivery day is in ``train_days``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epf.data import EXOG, TARGET
from epf.features import daily_matrix, feature_columns, make_features


class Forecaster:
    name = "base"

    def prepare(self, df: pd.DataFrame) -> None:
        self.df = df
        self.P = daily_matrix(df[TARGET])

    def fit(self, train_days: pd.DatetimeIndex) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, day: pd.Timestamp) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------- baselines
class NaiveWeekly(Forecaster):
    """Standard EPF naive benchmark: D-7 for Mon/Sat/Sun, D-1 otherwise."""

    name = "naive"

    def fit(self, train_days):
        pass

    def predict(self, day):
        k = 7 if day.dayofweek in (0, 5, 6) else 1
        return self.P.loc[day - pd.Timedelta(days=k)].to_numpy(dtype=float)


# ----------------------------------------------------------------- variance-stabilisation
class AsinhScaler:
    """Median/MAD normalisation followed by asinh (Uniejewski et al., 2018).

    Robust to price spikes and well defined for negative prices, unlike a log transform.
    """

    def fit(self, y: np.ndarray) -> AsinhScaler:
        self.med = float(np.nanmedian(y))
        self.mad = float(np.nanmedian(np.abs(y - self.med))) * 1.4826 or 1.0
        return self

    def transform(self, y):
        return np.arcsinh((y - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(z) * self.mad + self.med


# ------------------------------------------------------------------ gradient boosting
class LightGBMForecaster(Forecaster):
    name = "lgbm"

    def __init__(self, **params):
        self.params = {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31,
                           "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
                           "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1, "random_state": 0}
        self.params.update(params)

    def prepare(self, df):
        super().prepare(df)
        self.long = make_features(df)
        self.cols = feature_columns(self.long)

    def fit(self, train_days):
        import lightgbm as lgb

        rows = self.long.loc[self.long.index.get_level_values("day").isin(train_days)]
        rows = rows.dropna(subset=["y", "p_d7", "p_d7_mean"])
        target = (rows["y"] - rows["p_d1_mean"]).to_numpy()  # deviation from yesterday's level
        self.scaler = AsinhScaler().fit(target)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(rows[self.cols], self.scaler.transform(target))

    def predict(self, day):
        X = self.long.loc[day, self.cols]
        return self.scaler.inverse(self.model.predict(X)) + X["p_d1_mean"].to_numpy()

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.booster_.feature_importance("gain"),
                         index=self.cols).sort_values(ascending=False)


# ----------------------------------------------------------------------------- N-BEATS
class NBEATSForecaster(Forecaster):
    """N-BEATS (Oreshkin et al., 2020) with exogenous inputs, in the spirit of NBEATSx
    (Olivares et al., 2023): the block input is [price window | exog of D | calendar];
    only the price window is explained away by the backcast.
    """

    name = "nbeats"

    def __init__(self, lookback_days=7, n_blocks=4, hidden=256, layers=4, epochs=150,
                 batch_size=64, lr=1e-3, patience=15, seed=0):
        self.cfg = {"lookback_days": lookback_days, "n_blocks": n_blocks, "hidden": hidden,
                        "layers": layers, "epochs": epochs, "batch_size": batch_size, "lr": lr,
                        "patience": patience, "seed": seed}

    def prepare(self, df):
        super().prepare(df)
        self.X = {c: daily_matrix(df[c]) for c in EXOG}

    def _inputs(self, days):
        L = self.cfg["lookback_days"]
        pos = self.P.index.get_indexer(days)
        P = self.P.to_numpy(dtype=float)
        win = np.stack([P[p - L:p].ravel() for p in pos])  # days D-L..D-1
        level = win[:, -24:].mean(1, keepdims=True)  # D-1 mean: local level normalisation
        ex = np.concatenate([self.xs[c].transform(self.X[c].to_numpy()[pos]) for c in EXOG], 1)
        dow = np.eye(7)[pd.DatetimeIndex(days).dayofweek]
        return self.ys.transform(win - level), ex, dow, level

    def fit(self, train_days):
        import torch

        # torch and LightGBM ship separate OpenMP runtimes; on macOS both active in one
        # process deadlock. The network is small, so a single torch thread costs little.
        torch.set_num_threads(1)
        torch.manual_seed(self.cfg["seed"])
        L = self.cfg["lookback_days"]
        days = pd.DatetimeIndex([d for d in train_days if self.P.index.get_loc(d) >= L])
        days = days[~np.isnan(self.P.loc[days].to_numpy()).any(1)]
        pos = self.P.index.get_indexer(days)
        level = self.P.to_numpy(dtype=float)[pos - 1].mean(1, keepdims=True)
        y_rel = self.P.loc[days].to_numpy(dtype=float) - level
        self.ys = AsinhScaler().fit(y_rel.ravel())
        self.xs = {c: _StdScaler().fit(self.X[c].loc[days].to_numpy()) for c in EXOG}
        win, ex, dow, _ = self._inputs(days)
        y = self.ys.transform(y_rel)

        n_val = max(int(0.1 * len(days)), 14)  # most recent days for early stopping
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        tr = [t(a[:-n_val]) for a in (win, ex, dow, y)]
        va = [t(a[-n_val:]) for a in (win, ex, dow, y)]

        self.net = _NBEATSNet(win.shape[1], ex.shape[1] + 7, 24, self.cfg["n_blocks"],
                              self.cfg["hidden"], self.cfg["layers"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.cfg["epochs"]):
            self.net.train()
            perm = torch.randperm(len(tr[0]))
            for i in range(0, len(perm), self.cfg["batch_size"]):
                b = perm[i:i + self.cfg["batch_size"]]
                loss = torch.nn.functional.l1_loss(
                    self.net(tr[0][b], torch.cat([tr[1][b], tr[2][b]], 1)), tr[3][b])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = torch.nn.functional.l1_loss(
                    self.net(va[0], torch.cat([va[1], va[2]], 1)), va[3]).item()
            if v < best - 1e-4:
                best, wait = v, 0
                best_state = {k: x.clone() for k, x in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)

    def predict(self, day):
        import torch

        win, ex, dow, level = self._inputs(pd.DatetimeIndex([day]))
        with torch.no_grad():
            z = self.net(torch.tensor(win, dtype=torch.float32),
                         torch.tensor(np.concatenate([ex, dow], 1), dtype=torch.float32))
        return self.ys.inverse(z.numpy()[0]) + level[0, 0]


class _StdScaler:
    def fit(self, a):
        self.m, self.s = float(np.nanmean(a)), float(np.nanstd(a)) or 1.0
        return self

    def transform(self, a):
        return (a - self.m) / self.s


def _NBEATSNet(n_back, n_exog, n_fore, n_blocks, hidden, layers):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            mods, d = [], n_back + n_exog
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.ReLU()]
                d = hidden
            self.fc = nn.Sequential(*mods)
            self.back = nn.Linear(hidden, n_back)
            self.fore = nn.Linear(hidden, n_fore)

        def forward(self, x, e):
            h = self.fc(torch.cat([x, e], 1))
            return self.back(h), self.fore(h)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))

        def forward(self, x, e):
            forecast = 0
            for blk in self.blocks:  # doubly residual stacking
                b, f = blk(x, e)
                x = x - b
                forecast = forecast + f
            return forecast

    return Net()


REGISTRY = {"naive": NaiveWeekly, "lgbm": LightGBMForecaster, "nbeats": NBEATSForecaster}


def get_model(name: str, **kwargs) -> Forecaster:
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}") from None
