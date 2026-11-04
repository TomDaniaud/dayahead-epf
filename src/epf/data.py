"""Data loading and cleaning.

Canonical format used everywhere in the library: an hourly DataFrame indexed by a
tz-naive local-time DatetimeIndex with exactly 24 rows per delivery day and columns

    price      day-ahead clearing price (EUR/MWh)              -- target
    load_fc    day-ahead load forecast (MW)                    -- exogenous, known at D-1
    wind_fc    day-ahead wind generation forecast (MW)         -- exogenous, known at D-1
    solar_fc   day-ahead solar generation forecast (MW)        -- exogenous, known at D-1

Exogenous columns are *forecasts published before gate closure* (12:00 CET on D-1),
so using their values for day D is not a leak. Realised load/generation would be.
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

TARGET = "price"
EXOG = ["load_fc", "wind_fc", "solar_fc"]
MARKET_TZ = "Europe/Brussels"


def to_hourly_24(df: pd.DataFrame, tz: str = MARKET_TZ) -> pd.DataFrame:
    """Normalise any (tz-aware or UTC) series to 24 local hours per day.

    * resamples sub-hourly data to hourly means -- since 1 Oct 2025 the SDAC day-ahead
      auction clears in 15-minute MTUs;
    * converts to local market time and handles DST: the missing spring hour is
      interpolated and the duplicated autumn hour is averaged.
    """
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert("UTC").resample("1h").mean()
    local = df.tz_convert(tz)
    local.index = local.index.tz_localize(None)  # wall-clock time
    local = local.groupby(level=0).mean()  # autumn DST: average the duplicated hour
    full = pd.date_range(local.index.min().normalize(),
                         local.index.max().normalize() + pd.Timedelta(hours=23), freq="1h")
    local = local.reindex(full).interpolate(limit=3, limit_direction="both")
    # keep only days with all 24 hours present after short-gap interpolation
    bad_days = local.index.normalize()[local.isna().any(axis=1)].unique()
    return local[~local.index.normalize().isin(bad_days)]


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV previously written by :func:`download_entsoe` or :func:`synthetic_market`."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = {TARGET, *EXOG} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return df[[TARGET, *EXOG]]


def _fetch_period(api_key: str, country: str, a: pd.Timestamp, b: pd.Timestamp,
                  attempts: int = 3, retry_wait: float = 30.0) -> pd.DataFrame:
    """Download one chunk, retrying on network errors. Each call has its own client/session."""
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key, timeout=120)
    for attempt in range(1, attempts + 1):
        try:
            price = client.query_day_ahead_prices(country, start=a, end=b).rename(TARGET)
            load = client.query_load_forecast(country, start=a, end=b).iloc[:, 0].rename("load_fc")
            ws = client.query_wind_and_solar_forecast(country, start=a, end=b)
            break
        except OSError as e:  # requests' exceptions, timeouts included, derive from OSError
            if attempt == attempts:
                raise
            print(f"{a.date()}: {type(e).__name__}, retry {attempt}/{attempts - 1}", flush=True)
            time.sleep(retry_wait * attempt)
    wind = ws.filter(like="Wind").sum(axis=1).rename("wind_fc")
    solar = ws.get("Solar", pd.Series(0.0, index=ws.index)).rename("solar_fc")
    part = pd.concat([to_hourly_24(s.to_frame()) for s in (price, load, wind, solar)], axis=1)
    print(f"{a.date()} to {b.date()}: done", flush=True)
    return part


def download_entsoe(country: str, start: str, end: str, api_key: str | None = None,
                    workers: int = 8) -> pd.DataFrame:
    """Download prices + day-ahead load/wind/solar forecasts from the ENTSO-E Transparency Platform.

    Requires ``pip install dayahead-epf[entsoe]`` and a free API token
    (env var ``ENTSOE_API_KEY``; request it by email after registering on transparency.entsoe.eu).
    Queries are chunked by year and the years fetched concurrently: entsoe-py splits load and
    wind/solar into monthly requests and each one takes ~20 s server-side, so a sequential run
    is slow. Eight workers stay far below the platform's limit of 400 requests per minute.
    """
    api_key = api_key or os.environ["ENTSOE_API_KEY"]
    bounds = pd.date_range(pd.Timestamp(start, tz=MARKET_TZ),
                           pd.Timestamp(end, tz=MARKET_TZ), freq="YS").tolist()
    bounds = [pd.Timestamp(start, tz=MARKET_TZ), *bounds[1:], pd.Timestamp(end, tz=MARKET_TZ)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_period, api_key, country, a, b)
                   for a, b in itertools.pairwise(bounds)]
        parts = [f.result() for f in futures]  # submission order, so years stay sorted
    df = pd.concat(parts)
    return df[~df.index.duplicated()].dropna()


def synthetic_market(start: str = "2021-01-01", days: int = 1100, shift_day: int | None = 750,
                     seed: int = 0) -> pd.DataFrame:
    """Plausible synthetic market for tests and demos (no API key needed).

    Price = merit-order function of residual load (load - wind - solar) scaled by a
    fuel-cost random walk, plus spikes and negative prices on sunny/windy hours.
    At ``shift_day`` a regime change (fuel price jump + steeper merit order) mimics a
    gas-crisis-like break so that drift detection has something to find.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="1h")
    h = idx.hour.values
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values
    n = len(idx)

    season = np.cos(2 * np.pi * (doy - 15) / 365.25)  # +1 mid-January, -1 mid-July
    daily = 0.5 * (1 - np.cos(2 * np.pi * (h - 4) / 24)) + 0.25 * np.exp(-((h - 19) ** 2) / 4)
    load = 50_000 + 12_000 * season + 10_000 * daily - 6_000 * (dow >= 5)
    load += rng.normal(0, 1_500, n)

    wind = np.zeros(n)
    for t in range(1, n):  # AR(1) wind
        wind[t] = 0.97 * wind[t - 1] + rng.normal(0, 1_300)
    wind = np.clip(9_000 + 3_000 * season + wind, 300, None)

    sun = np.clip(np.sin(np.pi * (h - 6) / 14), 0, None) * (h >= 6) * (h <= 20)
    solar = 14_000 * sun * (0.75 - 0.35 * season) * rng.uniform(0.15, 1.0, days).repeat(24)

    fuel = np.cumsum(rng.normal(0, 0.6, days)).repeat(24) + 30.0
    steep = np.ones(n)
    if shift_day is not None:
        cut = shift_day * 24
        fuel[cut:] += 55.0
        steep[cut:] = 1.6

    residual = (load - wind - solar) / 1_000
    price = 0.9 * fuel + steep * (0.0035 * np.maximum(residual - 20, 0) ** 2.2) - 15
    price += rng.normal(0, 5, n)
    spikes = rng.random(n) < 0.004
    price[spikes] += rng.exponential(120, spikes.sum())
    price = np.where(residual < 18, price - (18 - residual) * 6, price)  # renewables glut

    return pd.DataFrame({TARGET: price, "load_fc": load * (1 + rng.normal(0, 0.02, n)),
                         "wind_fc": wind * (1 + rng.normal(0, 0.08, n)),
                         "solar_fc": solar * (1 + rng.normal(0, 0.05, n))}, index=idx)
