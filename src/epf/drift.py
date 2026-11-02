"""Drift detection.

* **Concept / performance drift** -- :class:`PageHinkley` on the stream of daily MAE
  normalised by the error level observed right after the last retraining. This is the
  signal that triggers retraining: the model got worse, whatever the cause.
* **Data drift** -- :func:`psi` (Population Stability Index) compares the recent
  distribution of an input feature to the training distribution. It is reported as a
  diagnostic (it can fire on harmless seasonal shifts, e.g. solar in summer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageHinkley:
    """Page-Hinkley test for an increase in the mean of a stream.

    m_t = sum_{i<=t} (x_i - mean_i - delta),  alarm when m_t - min_{s<=t} m_s > threshold.

    delta: tolerated drift magnitude (in units of x); threshold: alarm level (larger ->
    fewer false alarms, slower detection). ``warmup`` observations are seen before any
    alarm can fire.
    """

    delta: float = 0.1
    threshold: float = 4.0
    warmup: int = 14
    _n: int = field(default=0, init=False)
    _mean: float = field(default=0.0, init=False)
    _m: float = field(default=0.0, init=False)
    _min: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._n, self._mean, self._m, self._min = 0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self._n += 1
        self._mean += (x - self._mean) / self._n
        self._m += x - self._mean - self.delta
        self._min = min(self._min, self._m)
        return self._n > self.warmup and (self._m - self._min) > self.threshold

    @property
    def statistic(self) -> float:
        return self._m - self._min


class PerformanceDriftMonitor:
    """Feeds daily MAE, normalised by a reference error level, into Page-Hinkley.

    The reference is the median daily MAE of the first ``ref_days`` after a retrain
    (a robust, post-deployment estimate of "normal" error).
    """

    def __init__(self, ref_days: int = 14, delta: float = 0.1, threshold: float = 4.0):
        self.ref_days = ref_days
        self.ph = PageHinkley(delta=delta, threshold=threshold, warmup=ref_days)
        self.reset()

    def reset(self) -> None:
        self.ph.reset()
        self._ref: list[float] = []
        self.scale: float | None = None

    def update(self, daily_mae: float) -> bool:
        if self.scale is None:
            self._ref.append(daily_mae)
            if len(self._ref) == self.ref_days:
                self.scale = float(np.median(self._ref)) or 1.0
                for v in self._ref:
                    self.ph.update(v / self.scale)
            return False
        return self.ph.update(daily_mae / self.scale)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 shift."""
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))
