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
    # Sensitive to gentle palm-rest taps and table thumps (default 0.003)
    amplitude_threshold: float = 0.003

    # ── Spectral Discriminator (Mechanical Shock vs. Speaker Audio) ────
    # Ratio of structure-borne energy (25-350 Hz) to acoustic speaker energy (900-5000 Hz)
    # Physical knocks on chassis/desk have ratio > 1.2; speaker music/voice has ratio < 0.3
    min_mechanical_ratio: float = 0.65

    # ── Transient shape ───────────────────────────────────────────────
    # How fast the peak must rise relative to previous baseline
    rise_ratio_threshold: float = 1.6

    # ── Duration gate ─────────────────────────────────────────────────
    # A knock must stay above threshold for at least this long …
    min_duration_s: float = 0.005   # 5 ms (filters out tiny electrical pops)
    # … but MUST NOT be longer than this!
    # Sounds lasting > 160 ms (speech, claps, chair drag, continuous audio) are rejected.
    max_duration_s: float = 0.160   # 160 ms

    # ── Refractory period ─────────────────────────────────────────────
    # Minimum gap between two consecutive knock events (prevents echo double-trigger)
    refractory_s: float = 0.100     # 100 ms

    # ── Noise floor adaptation ────────────────────────────────────────
    # Running-average time constant for background noise estimate
    noise_floor_alpha: float = 0.985
    # Minimum multiplier above noise floor to call something a knock
    snr_multiplier: float = 3.0

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
