"""
knocklock/config.py
-------------------
Central configuration for KnockLock Phase 1.
All thresholds and audio settings live here so nothing is hard-coded
in the detection logic.
"""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Microphone / stream settings."""
    sample_rate: int = 44100        # Hz – standard mic rate
    channels: int = 1               # mono (or mixed down)
    chunk_size: int = 512           # frames per callback (≈11.6 ms @ 44100 Hz)
    dtype: str = "float32"          # PyAudio / sounddevice format
    device: object = None           # device index or name (None = default)


@dataclass
class FilterConfig:
    """Signal-processing parameters."""
    # High-pass filter to remove DC offset / slow air-con rumble
    # Set to 40 Hz so chassis knock thump (50-150 Hz) is NOT attenuated
    highpass_cutoff_hz: float = 40.0
    highpass_order: int = 2

    # Short-time energy window (seconds) used for RMS calculation
    energy_window_s: float = 0.010  # 10 ms


@dataclass
class DetectionConfig:
    """Knock-event detection thresholds."""
    # ── Amplitude gate (peak absolute value [0.0–1.0]) ───────────────
    # A physical tap on laptop or desk typically peaks between 0.01 and 0.10
    amplitude_threshold: float = 0.015

    # ── Transient shape ───────────────────────────────────────────────
    # How fast the peak must rise relative to previous baseline
    rise_ratio_threshold: float = 2.0

    # ── Duration gate ─────────────────────────────────────────────────
    # A knock must stay above threshold for at least this long …
    min_duration_s: float = 0.005   # 5 ms (filters out tiny electrical pops)
    # … but not longer than this (longer sounds are speech/rumble, NOT knocks)
    max_duration_s: float = 0.250   # 250 ms

    # ── Refractory period ─────────────────────────────────────────────
    # Minimum gap between two consecutive knock events (prevents echo double-trigger)
    refractory_s: float = 0.120     # 120 ms

    # ── Noise floor adaptation ────────────────────────────────────────
    # Running-average time constant for background noise estimate
    noise_floor_alpha: float = 0.990
    # Minimum multiplier above noise floor to call something a knock
    snr_multiplier: float = 3.0


@dataclass
class VisualizationConfig:
    """Debug / visualization settings."""
    enabled: bool = True
    # Width of the ASCII level-meter bar (characters)
    bar_width: int = 40
    # How many audio chunks to skip between display refreshes (reduces I/O)
    refresh_every_n_chunks: int = 2


@dataclass
class KnockLockConfig:
    """Top-level config object — pass this around instead of globals."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ── Convenience singleton ──────────────────────────────────────────────────
# Import this in other modules:  from knocklock.config import DEFAULT_CONFIG
DEFAULT_CONFIG = KnockLockConfig()
