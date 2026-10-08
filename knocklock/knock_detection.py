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
    compute_spectral_flatness,
    compute_crest_factor,
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
        self._knock_max_flatness: float = 0.0
        self._knock_max_crest: float = 0.0
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

        # 2. Compute energy, crest factor, and spectral flatness
        peak = compute_peak(filtered)
        rms = compute_rms(filtered)
        crest = peak / max(rms, 1e-6)
        flatness = compute_spectral_flatness(filtered)

        # 3. Dynamic effective threshold based on adaptive noise floor
        effective_threshold = max(
            det.amplitude_threshold,
            self._noise_floor.floor * det.snr_multiplier,
        )

        # Update noise floor during quiet non-knock periods
        if not self._in_knock and peak < effective_threshold:
            self._noise_floor.update(rms)

        # 4. Check whether chunk amplitude exceeds effective threshold
        above_threshold = peak >= effective_threshold

        # 5. Rise-ratio check relative to previous peak or noise floor
        baseline = max(self._prev_peak, self._noise_floor.floor, 1e-6)
        rise_ratio = peak / baseline

        refractory_elapsed = timestamp - self._last_knock_ts
        in_refractory = refractory_elapsed < det.refractory_s

        # ── State machine transitions ──────────────────────────────────
        if not self._in_knock:
            # Impulsive attack check:
            # - Knocks concentrate energy in a tiny spike (crest >= min_crest_factor).
            # - If peak >= 0.65, mic is clipping, which flattens the peak and reduces crest factor.
            is_impulsive = (crest >= det.min_crest_factor or peak >= 0.65)

            # Spectral check:
            # - Physical knock impacts are wideband transients (flatness >= min_spectral_flatness).
            # - Speaker music, dialogue, and video sounds are tonal/harmonic (flatness is low).
            is_broadband = flatness >= det.min_spectral_flatness

            is_onset = (
                above_threshold
                and not in_refractory
                and rise_ratio >= det.rise_ratio_threshold
                and is_impulsive
                and is_broadband
            )
            if is_onset:
                self._in_knock = True
                self._knock_start_ts = timestamp
                self._knock_peak_amp = peak
                self._knock_peak_rms = rms
                self._knock_duration_s = chunk_duration_s
                self._knock_max_flatness = flatness
                self._knock_max_crest = crest
            elif det.debug and above_threshold and not in_refractory:
                reasons = []
                if rise_ratio < det.rise_ratio_threshold:
                    reasons.append(f"slow rise ({rise_ratio:.1f}x < {det.rise_ratio_threshold}x)")
                if not is_impulsive:
                    reasons.append(f"low crest ({crest:.2f} < {det.min_crest_factor})")
                if not is_broadband:
                    reasons.append(f"tonal/video audio flatness ({flatness:.4f} < {det.min_spectral_flatness})")
                print(f"  [DEBUG REJECT ONSET] peak={peak:.3f} | {', '.join(reasons)}")
        else:
            # Inside candidate knock:
            if above_threshold:
                self._knock_duration_s += chunk_duration_s
                if peak > self._knock_peak_amp:
                    self._knock_peak_amp = peak
                if rms > self._knock_peak_rms:
                    self._knock_peak_rms = rms
                if flatness > self._knock_max_flatness:
                    self._knock_max_flatness = flatness
                if crest > self._knock_max_crest:
                    self._knock_max_crest = crest

                # Reject continuous sounds (speech, music, fans lasting > max_duration_s)
                if self._knock_duration_s > det.max_duration_s:
                    if det.debug:
                        print(f"  [DEBUG REJECT SUSTAINED] Sound > {det.max_duration_s*1000:.0f}ms (continuous speech/video)")
                    self._in_knock = False
                    self._knock_duration_s = 0.0
                    self._knock_peak_amp = 0.0
                    self._knock_peak_rms = 0.0
                    self._knock_max_flatness = 0.0

            # Sound has decayed back below threshold within valid knock window
            else:
                duration_ms = self._knock_duration_s * 1000.0

                if (
                    det.min_duration_s <= self._knock_duration_s <= det.max_duration_s
                    and self._knock_max_flatness >= det.min_spectral_flatness
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
                self._knock_max_flatness = 0.0

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
