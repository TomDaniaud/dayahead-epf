import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest

from epf.backtest import BacktestConfig, walk_forward
from epf.data import download_entsoe, synthetic_market, to_hourly_24
from epf.drift import PageHinkley, PerformanceDriftMonitor, psi
from epf.features import daily_matrix, make_features
from epf.metrics import diebold_mariano, forecast_kpis
from epf.models import AsinhScaler, NaiveWeekly, get_model


@pytest.fixture(scope="module")
def market():
    return synthetic_market(days=500, shift_day=380, seed=1)


# ------------------------------------------------------------------ data & features
class FakeEntsoeClient:
    """Fails once per period with a timeout, then serves hourly data."""
    failed: set = set()
    lock = threading.Lock()

    def __init__(self, api_key, timeout=None):
        pass

    @staticmethod
    def _index(start, end):
        return pd.date_range(start, end, freq="1h", tz="Europe/Brussels", inclusive="left")

    def query_day_ahead_prices(self, country, start, end):
        with self.lock:
            if start not in self.failed:
                self.failed.add(start)
                raise TimeoutError("read timed out")
        idx = self._index(start, end)
        return pd.Series(float(start.year), index=idx)

    def query_load_forecast(self, country, start, end):
        return pd.DataFrame({"Forecasted Load": 1.0}, index=self._index(start, end))

    def query_wind_and_solar_forecast(self, country, start, end):
        return pd.DataFrame({"Solar": 1.0, "Wind Onshore": 1.0, "Wind Offshore": 1.0},
                            index=self._index(start, end))


def test_download_keeps_years_in_order_and_retries(monkeypatch):
    monkeypatch.setitem(sys.modules, "entsoe", types.SimpleNamespace(EntsoePandasClient=FakeEntsoeClient))
    monkeypatch.setattr("epf.data.time.sleep", lambda s: None)
    df = download_entsoe("FR", "2022-01-01", "2024-07-01", api_key="x")
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert sorted(df["price"].unique()) == [2022.0, 2023.0, 2024.0]
    assert df.index.min() == pd.Timestamp("2022-01-01")
    assert (df.groupby(df.index.normalize()).size() == 24).all()
    assert (df["wind_fc"] == 2.0).all()
    assert len(FakeEntsoeClient.failed) == 3  # each year failed once, then recovered


def test_dst_handling_gives_24_hours_per_day():
    idx = pd.date_range("2024-03-30", "2024-11-01", freq="15min", tz="UTC")
    df = pd.DataFrame({"price": np.arange(len(idx), dtype=float)}, index=idx)
    out = to_hourly_24(df)
    assert (out.groupby(out.index.normalize()).size() == 24).all()
    assert out.index.tz is None


def test_no_leakage(market):
    """Changing prices on day D must not change any feature of day D."""
    day = market.index.normalize().unique()[100]
    perturbed = market.copy()
    perturbed.loc[perturbed.index.normalize() == day, "price"] += 1_000
    a = make_features(market).loc[day].drop(columns="y")
    b = make_features(perturbed).loc[day].drop(columns="y")
    pd.testing.assert_frame_equal(a, b)


def test_asinh_scaler_roundtrip():
    y = np.array([-50.0, 0.0, 30.0, 80.0, 900.0])
    s = AsinhScaler().fit(y)
    np.testing.assert_allclose(s.inverse(s.transform(y)), y)


# ------------------------------------------------------------------------- metrics
def test_kpis_perfect_forecast():
    y = pd.DataFrame(np.random.default_rng(0).normal(50, 10, (10, 24)))
    k = forecast_kpis(y, y, y + 5)
    assert k["MAE"] == 0 and k["RMSE"] == 0 and k["rMAE"] == 0


def test_diebold_mariano_detects_better_forecast():
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.normal(50, 10, (300, 24)))
    good = y + rng.normal(0, 1, y.shape)
    bad = y + rng.normal(0, 5, y.shape)
    assert diebold_mariano(y, bad, good)["p_value"] < 0.01
    assert diebold_mariano(y, good, bad)["p_value"] > 0.99


# --------------------------------------------------------------------------- drift
def test_page_hinkley_no_alarm_on_stationary_stream():
    ph = PageHinkley(delta=0.1, threshold=4.0)
    rng = np.random.default_rng(0)
    assert not any(ph.update(x) for x in rng.normal(1, 0.2, 500))


def test_page_hinkley_detects_mean_shift():
    ph = PageHinkley(delta=0.1, threshold=4.0)
    rng = np.random.default_rng(0)
    stream = np.r_[rng.normal(1, 0.2, 100), rng.normal(2.5, 0.2, 50)]
    alarms = [i for i, x in enumerate(stream) if ph.update(x)]
    assert alarms and 100 <= alarms[0] < 110


def test_monitor_normalises_by_reference():
    m = PerformanceDriftMonitor(ref_days=10)
    assert not any(m.update(100.0) for _ in range(30))  # high but stable error: no alarm
    assert any(m.update(300.0) for _ in range(10))


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25


# ----------------------------------------------------------------- models/backtest
def _with_gap(market, first="2021-03-10", last="2021-03-11"):
    days = market.index.normalize()
    return market[(days < pd.Timestamp(first)) | (days > pd.Timestamp(last))]


def test_lags_follow_the_calendar_across_missing_days(market):
    df = _with_gap(market)
    features = make_features(df)
    day = pd.Timestamp("2021-03-12")  # first day after the gap
    level = df.loc["2021-03-09", "price"].mean()  # last known prices stand in for D-1
    assert features.loc[(day, 0), "p_d1_mean"] == pytest.approx(level)
    d7 = market.loc["2021-03-05 00:00", "price"]
    assert features.loc[(day, 0), "p_d7"] == pytest.approx(d7 - level)


def test_filled_days_only_use_the_past(market):
    df = _with_gap(market)
    filled = daily_matrix(df["price"]).loc["2021-03-10"].to_numpy()
    assert np.array_equal(filled, df.loc["2021-03-09", "price"].to_numpy())


@pytest.mark.parametrize("name", ["naive", "lgbm"])
def test_backtest_skips_missing_days_without_crashing(market, name):
    df = _with_gap(market)
    cfg = BacktestConfig(test_start="2021-03-05", test_end="2021-03-20", retrain="never")
    r = walk_forward(get_model(name), df, cfg)
    missing = pd.to_datetime(["2021-03-10", "2021-03-11"])
    assert not r.forecasts.index.isin(missing).any()
    assert len(r.forecasts) == 14
    assert not r.forecasts.isna().any().any()


def test_naive_is_d7_on_monday(market):
    m = NaiveWeekly()
    m.prepare(market)
    days = m.P.index
    monday = days[(days.dayofweek == 0)][5]
    np.testing.assert_array_equal(m.predict(monday), m.P.loc[monday - pd.Timedelta(days=7)])


@pytest.mark.parametrize("policy", ["never", "periodic", "drift", "periodic+drift"])
def test_backtest_runs_and_lgbm_beats_naive(market, policy):
    cfg = BacktestConfig(test_start="2021-12-01", retrain=policy, window_days=300)
    r = walk_forward(get_model("lgbm", n_estimators=200), market, cfg)
    assert r.forecasts.shape == r.actuals.shape
    assert np.isfinite(r.forecasts.to_numpy()).all()
    assert r.kpis["rMAE"] < 1.0


def test_drift_retrain_fires_after_regime_shift(market):
    cfg = BacktestConfig(test_start="2021-12-01", retrain="drift", window_days=300)
    r = walk_forward(get_model("lgbm", n_estimators=200), market, cfg)
    shift = pd.Timestamp("2021-01-01") + pd.Timedelta(days=380)
    drift_days = [e["day"] for e in r.retrains if e["reason"] == "drift"]
    assert any(shift <= d <= shift + pd.Timedelta(days=10) for d in drift_days)


def test_training_never_sees_test_day(market):
    seen = []

    class Spy(NaiveWeekly):
        def fit(self, train_days):
            seen.append(train_days.max())

    cfg = BacktestConfig(test_start="2021-12-01", retrain="periodic", retrain_every=20)
    r = walk_forward(Spy(), market, cfg)
    for e, last_train_day in zip(r.retrains, seen):
        assert last_train_day < e["day"]


def test_nbeats_smoke(market):
    torch = pytest.importorskip("torch")  # noqa: F841
    cfg = BacktestConfig(test_start="2022-04-01", test_end="2022-04-20", retrain="never",
                         window_days=200)
    r = walk_forward(get_model("nbeats", epochs=5, hidden=32, n_blocks=2), market, cfg)
    assert np.isfinite(r.forecasts.to_numpy()).all()
