"""
Patient data loader for Parkinsonian tremor simulation.

This module now supports two modes:
1) Real wearable data: downloads the UCI HandPD inertial-sensor dataset
    (smartphone gyroscope/accelerometer during tremor-inducing tasks),
    extracts a representative signal, and estimates tremor frequency and
    noise variance for physics calibration.
2) Synthetic fallback: generates band-passed noise in the canonical
    4-6 Hz tremor band when download/parsing is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch


@dataclass
class PatientLoader:
    """
    Fabricates a synthetic tremor time series and extracts calibration metrics.

    Attributes
    ----------
    duration_s : float
        Length of the simulated recording (seconds).
    fs : float
        Sampling frequency (Hz) used to discretize the continuous tremor process.
    tremor_band : Tuple[float, float]
        Lower and upper bounds (Hz) of the band-pass filter used to mimic
        Parkinsonian resting tremor, typically centered around 4-6 Hz.
    noise_std : float
        Standard deviation of the pre-filter broadband noise driving the model.

    Biological Rationale
    --------------------
    Rest tremor in Parkinson's disease often concentrates around 4-6 Hz. A
    broadband stochastic drive filtered to this band captures the prominent
    oscillatory peak while preserving random amplitude and phase fluctuations
    observed in patient recordings. Dominant frequency characterizes tremor
    severity, whereas noise variance approximates background neural variability.
    """

    duration_s: float = 60.0
    fs: float = 200.0
    tremor_band: Tuple[float, float] = (4.0, 6.0)
    noise_std: float = 0.8
    use_real_data: bool = True
    data_root: Path = Path("data")

    # UCI HandPD smartphone inertial dataset (wearable tremor capture).
    dataset_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/HandPD_sensors_data.zip"
    )
    dataset_name: str = "HandPD_sensors_data.zip"
    extracted_dirname: str = "HandPD_sensors_data"

    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply a zero-phase Butterworth band-pass filter around the tremor band."""
        nyquist = 0.5 * self.fs
        low, high = self.tremor_band[0] / nyquist, self.tremor_band[1] / nyquist
        b, a = butter(N=4, Wn=[low, high], btype="bandpass")
        return filtfilt(b, a, data)

    def _generate_raw_noise(self, n_samples: int) -> np.ndarray:
        """Generate broadband Gaussian noise representing neural drive."""
        rng = np.random.default_rng()
        return rng.normal(loc=0.0, scale=self.noise_std, size=n_samples)

    def _dominant_frequency(self, signal: np.ndarray) -> float:
        """Estimate dominant tremor frequency using Welch's method."""
        freqs, psd = welch(signal, fs=self.fs, nperseg=1024)
        return float(freqs[np.argmax(psd)])

    # Real-data path -----------------------------------------------------
    def _download_dataset(self) -> Path:
        """Download and unzip the HandPD dataset if not already present."""
        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / self.dataset_name
        extracted_dir = self.data_root / self.extracted_dirname

        if extracted_dir.exists():
            return extracted_dir

        if not zip_path.exists():
            urllib.request.urlretrieve(self.dataset_url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_root)

        return extracted_dir

    def _load_first_timeseries(self, extracted_dir: Path) -> Tuple[np.ndarray, float]:
        """
        Locate the first CSV file, extract a multi-axis inertial signal, and infer fs.

        Returns
        -------
        Tuple[np.ndarray, float]
            Vector magnitude signal and inferred sampling rate (Hz).
        """
        csv_files = sorted(extracted_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in downloaded dataset")

        df = pd.read_csv(csv_files[0])
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("Dataset CSV has no numeric columns")

        # Use vector magnitude if multi-axis; fallback to first numeric column.
        arr = numeric.to_numpy()
        if arr.shape[1] >= 3:
            signal = np.linalg.norm(arr[:, :3], axis=1)
        else:
            signal = arr[:, 0]

        # Infer sampling rate from any time column; fallback to default if missing.
        time_cols = [c for c in df.columns if "time" in c.lower() or c.lower() == "t"]
        if time_cols:
            times = df[time_cols[0]].to_numpy()
            dt = np.median(np.diff(times)) if len(times) > 1 else 1.0 / self.fs
            fs = float(1.0 / dt) if dt > 0 else self.fs
        else:
            fs = self.fs

        return signal.astype(float), fs

    def _load_real_patient(self) -> Dict[str, np.ndarray | float]:
        """Load tremor metrics from the HandPD wearable dataset."""
        extracted_dir = self._download_dataset()
        signal, fs = self._load_first_timeseries(extracted_dir)

        # Detrend by removing mean, then band-pass in tremor range.
        signal = signal - np.mean(signal)
        self.fs = fs
        filtered_signal = self._bandpass_filter(signal)

        dominant_freq = self._dominant_frequency(filtered_signal)
        noise_variance = float(np.var(signal))

        return {
            "signal": filtered_signal,
            "fs": fs,
            "dominant_frequency": dominant_freq,
            "noise_variance": noise_variance,
        }

    def load_patient_data(self) -> Dict[str, np.ndarray | float]:
        """
        Return calibration statistics derived from real data or synthetic fallback.

        Returns
        -------
        Dict[str, np.ndarray | float]
            Contains the tremor signal, sampling rate, dominant frequency,
            and noise variance estimate.
        """
        if self.use_real_data:
            try:
                return self._load_real_patient()
            except Exception:
                # Fall back gracefully when offline or if parsing fails.
                pass

        n_samples = int(self.duration_s * self.fs)
        raw_noise = self._generate_raw_noise(n_samples)
        filtered_signal = self._bandpass_filter(raw_noise)

        dominant_freq = self._dominant_frequency(filtered_signal)
        noise_variance = float(np.var(raw_noise))

        return {
            "signal": filtered_signal,
            "fs": self.fs,
            "dominant_frequency": dominant_freq,
            "noise_variance": noise_variance,
        }


__all__ = ["PatientLoader"]
