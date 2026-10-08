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
    # 20 Hz high-pass removes DC offset while preserving low-frequency table thumps (25-80 Hz)
    highpass_cutoff_hz: float = 20.0
    highpass_order: int = 2

    # Short-time energy window (seconds) used for RMS calculation
    energy_window_s: float = 0.010  # 10 ms


@dataclass
class DetectionConfig:
    """Knock-event detection thresholds."""
    # ── Amplitude gate (peak absolute value [0.0–1.0]) ───────────────
    # Normal video/speech audio peaks at 0.004–0.025.
    # Physical knocks and table thumps peak at 0.05–0.99.
    amplitude_threshold: float = 0.035

    # ── Spectral Flatness (Wiener Entropy) ───────────────────────────
    # Distinguishes broadband mechanical impulses from tonal speaker audio (speech/music).
    # Real physical knocks/thumps have power flatness 0.20–0.70.
    # Video speech/music from laptop speakers has power flatness 0.001–0.020.
    min_spectral_flatness: float = 0.05

    # ── Crest Factor (Peak / RMS) ────────────────────────────────────
    # Physical impulses concentrate energy into sharp spikes (crest factor > 2.4).
    # Continuous speaker audio / speech has crest factor < 2.3.
    min_crest_factor: float = 2.4

    # Backwards-compatible mechanical ratio field
    min_mechanical_ratio: float = 0.55

    # ── Transient shape ───────────────────────────────────────────────
    # How fast the peak must rise relative to previous baseline
    rise_ratio_threshold: float = 1.8

    # ── Duration gate ─────────────────────────────────────────────────
    # A knock must stay above threshold for at least this long …
    min_duration_s: float = 0.005   # 5 ms
    # Maximum allowed duration for an impulse (longer sounds are continuous noise/speech/music)
    max_duration_s: float = 0.120   # 120 ms

    # ── Refractory period ─────────────────────────────────────────────
    # Minimum gap between two consecutive knock events (prevents echo/rebound double-trigger)
    refractory_s: float = 0.150     # 150 ms

    # ── Noise floor adaptation ────────────────────────────────────────
    # Running-average time constant for background noise estimate
    noise_floor_alpha: float = 0.985
    # Minimum multiplier above noise floor to call something a knock
    snr_multiplier: float = 2.8

    # ── Debug diagnostics ─────────────────────────────────────────────
    debug: bool = False


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
