"""
knocklock/signal_processing.py
-------------------------------
Low-level DSP helpers.

Responsibilities
----------------
* High-pass filtering to remove hum/rumble
* Short-time RMS (energy) computation
* Adaptive noise-floor estimation
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from knocklock.config import AudioConfig, FilterConfig, DetectionConfig


# ── High-pass filter ──────────────────────────────────────────────────────

def design_highpass(
    cutoff_hz: float, sample_rate: int, order: int = 4
) -> np.ndarray:
    """Return second-order-sections (SOS) for a Butterworth high-pass filter."""
    nyq = sample_rate / 2.0
    normal_cutoff = cutoff_hz / nyq
    sos = butter(order, normal_cutoff, btype="high", analog=False, output="sos")
    return sos


class HighPassFilter:
    """
    Stateful high-pass filter for streaming audio.

    The filter state (zi) is preserved across chunks so the filter is
    continuous across buffer boundaries.
    """

    def __init__(self, config_audio: AudioConfig, config_filter: FilterConfig):
        self.sos = design_highpass(
            config_filter.highpass_cutoff_hz,
            config_audio.sample_rate,
            config_filter.highpass_order,
        )
        # Initialise filter delay lines to zero
        self._zi = sosfilt_zi(self.sos)  # shape: (n_sections, 2)
        self._zi *= 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a 1-D audio chunk, returning a filtered copy."""
        filtered, self._zi = sosfilt(self.sos, chunk, zi=self._zi)
        return filtered


# ── Short-time energy (RMS) ───────────────────────────────────────────────

def compute_rms(chunk: np.ndarray) -> float:
    """Root-mean-square energy of a 1-D audio chunk (returns 0 for silence)."""
    if len(chunk) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return rms


def compute_peak(chunk: np.ndarray) -> float:
    """Absolute peak amplitude of a 1-D audio chunk."""
    if len(chunk) == 0:
        return 0.0
    return float(np.max(np.abs(chunk)))


# ── Adaptive noise floor ──────────────────────────────────────────────────

class NoiseFloorEstimator:
    """
    Tracks the long-term background noise level using an exponential
    moving average (EMA).

    The floor is only updated when the incoming RMS is *below* the
    current estimate × a small multiplier (so loud transients don't
    drag the floor upward).
    """

    def __init__(self, config: DetectionConfig, initial_value: float = 1e-4):
        self._alpha = config.noise_floor_alpha
        self._floor: float = initial_value
        self._snr_mult = config.snr_multiplier

    @property
    def floor(self) -> float:
        return self._floor

    def update(self, rms: float) -> None:
        """Feed a new RMS value; update the floor if it represents quiet."""
        if rms < self._floor * self._snr_mult:
            # This chunk is quiet — fold it into the running average
            self._floor = self._alpha * self._floor + (1 - self._alpha) * rms

    def is_above_floor(self, rms: float) -> bool:
        """True if rms exceeds the dynamic noise floor by the SNR multiplier."""
        return rms > self._floor * self._snr_mult


# ── Spectral Discriminator (Mechanical Shock vs. Speaker Audio) ───────────

def compute_mechanical_ratio(chunk: np.ndarray, sample_rate: int = 44100) -> float:
    """
    Ratio of structure-borne mechanical shock energy (25-350 Hz)
    to airborne acoustic speaker/vocal energy (900-5000 Hz).

    - Physical knock on laptop or desk: conducts mechanically through chassis,
      producing high low-frequency shock energy (ratio > 1.2 to 10.0).
    - Laptop speakers playing audio / human speech: dominated by mid/high
      frequencies, with tiny micro-speakers having severe bass roll-off (ratio < 0.4).
    """
    if len(chunk) < 64:
        return 2.0
    fft = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
    low_energy = float(np.sum(fft[(freqs >= 25.0) & (freqs <= 350.0)]))
    high_energy = float(np.sum(fft[(freqs >= 900.0) & (freqs <= 5000.0)]))
    if high_energy < 1e-9:
        return 5.0
    return low_energy / high_energy
