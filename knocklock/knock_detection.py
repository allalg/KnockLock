"""
knocklock/knock_detection.py
-----------------------------
Stateful knock-event detector.

Knock detection algorithm
--------------------------
Each audio chunk is processed in order:

1. Filter  → remove hum/rumble (HighPassFilter)
2. Energy  → compute RMS of the filtered chunk
3. Gate    → compare RMS to noise floor (NoiseFloorEstimator)
             AND to the hard amplitude threshold
4. Rise    → check that energy rose fast enough (transient shape)
5. Active  → track how long we stay above threshold (knock duration)
6. Emit    → when energy falls back below threshold, emit a KnockEvent
             if duration is within [min, max] window
7. Refractory → ignore new transients for a short period after a knock

This deliberately separates *detection* from *reporting* — the detector
emits KnockEvent objects; callers decide how to display/store them.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from knocklock.config import KnockLockConfig
from knocklock.signal_processing import (
    HighPassFilter,
    NoiseFloorEstimator,
    compute_rms,
    compute_peak,
    compute_mechanical_ratio,
)


@dataclass
class KnockEvent:
    """All information captured for a single detected knock."""
    index: int                          # sequential knock number (1-based)
    timestamp: float                    # seconds since capture started
    amplitude: float                    # peak normalised amplitude [0–1]
    rms: float                          # RMS energy at peak chunk
    duration_ms: float                  # how long the transient lasted (ms)
    noise_floor: float                  # background noise at detection time

    def __str__(self) -> str:
        return (
            f"Knock #{self.index:>3}  "
            f"t={self.timestamp:>7.3f}s  "
            f"amp={self.amplitude:.3f}  "
            f"rms={self.rms:.4f}  "
            f"dur={self.duration_ms:>5.1f}ms  "
            f"floor={self.noise_floor:.5f}"
        )


class KnockDetector:
    """
    Processes a continuous stream of (chunk, timestamp) pairs and calls
    `on_knock` whenever a knock event is confirmed.

    Parameters
    ----------
    config   : KnockLockConfig
    on_knock : callback(event: KnockEvent) — called on the calling thread.
    """

    def __init__(
        self,
        config: KnockLockConfig,
        on_knock: Optional[Callable[[KnockEvent], None]] = None,
    ):
        self.config = config
        self.on_knock = on_knock

        self._hp_filter = HighPassFilter(config.audio, config.filter)
        self._noise_floor = NoiseFloorEstimator(config.detection)

        # ── State machine ─────────────────────────────────────────────
        self._in_knock: bool = False
        self._knock_start_ts: float = 0.0
        self._knock_peak_amp: float = 0.0
        self._knock_peak_rms: float = 0.0
        self._knock_duration_s: float = 0.0
        self._knock_max_mech_ratio: float = 0.0
        self._prev_peak: float = 0.0
        self._prev_rms: float = 0.0

        # Refractory state
        self._last_knock_ts: float = -999.0

        self._knock_count: int = 0
        self._chunk_count: int = 0

    # ── Public ────────────────────────────────────────────────────────────

    def process(self, chunk: np.ndarray, timestamp: float) -> None:
        """
        Feed the next audio chunk into the detector.

        Parameters
        ----------
        chunk     : 1-D float32 array of audio samples
        timestamp : wall-clock time at the start of this chunk (seconds)
        """
        self._chunk_count += 1
        det = self.config.detection
        sr = self.config.audio.sample_rate
        chunk_duration_s = len(chunk) / sr if sr > 0 else 0.0116

        # 1. High-pass filter (20 Hz preserves table thumps down to 25 Hz)
        filtered = self._hp_filter.process(chunk)

        # 2. Compute energy and spectral mechanical shock metrics
        peak = compute_peak(filtered)
        rms = compute_rms(filtered)
        mech_ratio = compute_mechanical_ratio(filtered, sr)

        # 3. Dynamic effective threshold based on adaptive noise floor
        effective_threshold = max(
            det.amplitude_threshold,
            self._noise_floor.floor * det.snr_multiplier,
        )

        # Update noise floor during quiet non-knock periods
        if not self._in_knock and peak < effective_threshold:
            self._noise_floor.update(peak)

        # 4. Check whether chunk amplitude exceeds effective threshold
        above_threshold = peak >= effective_threshold

        # 5. Rise-ratio check relative to previous peak or noise floor
        baseline = max(self._prev_peak, self._noise_floor.floor, 1e-6)
        rise_ratio = peak / baseline

        refractory_elapsed = timestamp - self._last_knock_ts
        in_refractory = refractory_elapsed < det.refractory_s

        # ── State machine transitions ──────────────────────────────────
        if not self._in_knock:
            # Mechanical shock check: physical knocks on chassis/desk have high low-freq shock (>= min_mechanical_ratio)
            # Laptop speakers (music/video) and speech are dominated by mid/high freqs and fail this check
            is_mech_shock = mech_ratio >= det.min_mechanical_ratio

            is_onset = (
                above_threshold
                and not in_refractory
                and is_mech_shock
                and (
                    rise_ratio >= det.rise_ratio_threshold
                    or peak >= effective_threshold * 1.5
                )
            )
            if is_onset:
                self._in_knock = True
                self._knock_start_ts = timestamp
                self._knock_peak_amp = peak
                self._knock_peak_rms = rms
                self._knock_duration_s = chunk_duration_s
                self._knock_max_mech_ratio = mech_ratio
        else:
            # Inside candidate knock:
            if above_threshold:
                self._knock_duration_s += chunk_duration_s
                if peak > self._knock_peak_amp:
                    self._knock_peak_amp = peak
                if rms > self._knock_peak_rms:
                    self._knock_peak_rms = rms
                if mech_ratio > self._knock_max_mech_ratio:
                    self._knock_max_mech_ratio = mech_ratio

                # Reject continuous noise (speech, fans, long sustained sounds > max_duration_s)
                if self._knock_duration_s > det.max_duration_s:
                    self._in_knock = False
                    self._knock_duration_s = 0.0
                    self._knock_peak_amp = 0.0
                    self._knock_peak_rms = 0.0

            # Sound has dropped back below threshold within valid knock window
            else:
                duration_ms = self._knock_duration_s * 1000.0

                if (
                    det.min_duration_s <= self._knock_duration_s <= det.max_duration_s
                    and self._knock_max_mech_ratio >= det.min_mechanical_ratio
                ):
                    self._knock_count += 1
                    event = KnockEvent(
                        index=self._knock_count,
                        timestamp=self._knock_start_ts,
                        amplitude=self._knock_peak_amp,
                        rms=self._knock_peak_rms,
                        duration_ms=duration_ms,
                        noise_floor=self._noise_floor.floor,
                    )
                    self._last_knock_ts = timestamp
                    if self.on_knock:
                        self.on_knock(event)

                self._in_knock = False
                self._knock_peak_amp = 0.0
                self._knock_peak_rms = 0.0
                self._knock_duration_s = 0.0

        self._prev_peak = peak
        self._prev_rms = rms

    @property
    def knock_count(self) -> int:
        return self._knock_count

    @property
    def noise_floor(self) -> float:
        return self._noise_floor.floor

    @property
    def chunk_count(self) -> int:
        return self._chunk_count
