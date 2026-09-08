"""CSV-imported signal block (docs/CONCEPT.md §11.2/§11.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SignalDesign


class CsvSignal(SignalDesign):
    """Two-column CSV (time [s], value), linearly resampled onto the chain's
    `fs` grid at `generate()` time -- raw file values are not assumed
    pre-normalized (§11.2), so `factor` (applied by the caller) is the only
    scaling knob for this block kind.

    Values outside the file's own recorded time range are padded with 0.0
    rather than extrapolated or held at the last value -- CONCEPT.md §11.4
    leaves the exact edge behavior open; this is this iteration's choice.
    """

    kind = "CSV"

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._time: np.ndarray | None = None
        self._value: np.ndarray | None = None
        self.validate()

    def validate(self) -> None:
        if not self.file_path:
            raise ValueError("no CSV file selected")
        path = Path(self.file_path)
        if not path.is_file():
            raise ValueError(f"CSV file not found: {self.file_path}")
        data = self._load(path)
        if data.shape[1] < 2:
            raise ValueError(f"CSV file must have at least 2 columns (time, value), got {data.shape[1]}")
        if data.shape[0] < 2:
            raise ValueError(f"CSV file must have at least 2 rows, got {data.shape[0]}")
        order = np.argsort(data[:, 0], kind="stable")
        time = data[order, 0]
        value = data[order, 1]
        if np.any(np.diff(time) <= 0):
            raise ValueError("CSV time column must be strictly increasing (duplicate timestamps found)")
        self._time = time
        self._value = value

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        """Parses a 2+ column numeric CSV, tolerating an optional header row."""
        try:
            return np.loadtxt(path, delimiter=",", ndmin=2)
        except ValueError:
            try:
                return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            except ValueError as exc:
                raise ValueError(f"could not parse CSV file {path}: {exc}") from None
        except OSError as exc:
            raise ValueError(f"could not read CSV file {path}: {exc}") from None

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        t = np.arange(n_samples, dtype=np.float64) / fs
        return np.interp(t, self._time, self._value, left=0.0, right=0.0)
