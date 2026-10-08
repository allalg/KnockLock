"""
tests/test_signal_processing.py
--------------------------------
Unit tests for the signal-processing helpers.

Run with:  pytest tests/
"""

import numpy as np
import pytest

from knocklock.config import AudioConfig, FilterConfig, DetectionConfig
from knocklock.signal_processing import (
    HighPassFilter,
    NoiseFloorEstimator,
    compute_rms,
    compute_peak,
    compute_mechanical_ratio,
    compute_spectral_flatness,
    compute_crest_factor,
    design_highpass,
)


# ── compute_rms ───────────────────────────────────────────────────────────

class TestComputeRms:
    def test_silence(self):
        assert compute_rms(np.zeros(512, dtype=np.float32)) == 0.0

    def test_dc_signal(self):
        chunk = np.ones(512, dtype=np.float32) * 0.5
        assert abs(compute_rms(chunk) - 0.5) < 1e-6

    def test_sine(self):
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        # RMS of a full-scale sine is 1/sqrt(2) ≈ 0.7071
        assert abs(compute_rms(sine) - (1.0 / np.sqrt(2))) < 1e-3

    def test_empty(self):
        assert compute_rms(np.array([], dtype=np.float32)) == 0.0


# ── compute_peak ─────────────────────────────────────────────────────────

class TestComputePeak:
    def test_basic(self):
        chunk = np.array([-0.8, 0.3, 0.5], dtype=np.float32)
        assert abs(compute_peak(chunk) - 0.8) < 1e-6

    def test_empty(self):
        assert compute_peak(np.array([], dtype=np.float32)) == 0.0


# ── HighPassFilter ────────────────────────────────────────────────────────

class TestHighPassFilter:
    def _make_filter(self):
        return HighPassFilter(AudioConfig(), FilterConfig())

    def test_dc_removal(self):
        """A DC signal (frequency 0) should be attenuated strongly."""
        hp = self._make_filter()
        dc = np.ones(8192, dtype=np.float32) * 0.5
        out = hp.process(dc)
        # After many samples the filter should settle; check tail
        assert compute_rms(out[4096:]) < 0.05

    def test_high_freq_passes(self):
        """A 1 kHz tone should pass through mostly intact."""
        cfg_audio = AudioConfig()
        hp = HighPassFilter(cfg_audio, FilterConfig())
        sr = cfg_audio.sample_rate
        t = np.linspace(0, 1, sr, dtype=np.float32)
        tone = np.sin(2 * np.pi * 1000 * t)
        out = hp.process(tone)
        # Allow for some transient at the start; check steady-state
        rms_in = compute_rms(tone[sr // 4 :])
        rms_out = compute_rms(out[sr // 4 :])
        assert rms_out > rms_in * 0.85  # less than 15 % attenuation

    def test_stateful_continuity(self):
        """Filter state should be consistent across chunk boundaries."""
        hp = HighPassFilter(AudioConfig(), FilterConfig())
        sr = AudioConfig().sample_rate
        chunk_size = 512
        t_full = np.linspace(0, 1, sr, dtype=np.float32)
        signal = np.sin(2 * np.pi * 440 * t_full)

        # Process in chunks
        outputs = []
        for i in range(0, sr, chunk_size):
            outputs.append(hp.process(signal[i : i + chunk_size]))
        chunked = np.concatenate(outputs)

        # Process all at once (fresh filter)
        hp2 = HighPassFilter(AudioConfig(), FilterConfig())
        full = hp2.process(signal)

        np.testing.assert_allclose(chunked, full, atol=1e-5)


# ── NoiseFloorEstimator ───────────────────────────────────────────────────

class TestNoiseFloorEstimator:
    def _make(self) -> NoiseFloorEstimator:
        return NoiseFloorEstimator(DetectionConfig())

    def test_floor_falls_with_silence(self):
        est = self._make()
        initial = est.floor
        for _ in range(5000):
            est.update(1e-6)
        assert est.floor < initial

    def test_loud_transient_does_not_raise_floor(self):
        est = self._make()
        # Settle floor first
        for _ in range(1000):
            est.update(1e-6)
        floor_before = est.floor
        # Feed a loud burst
        for _ in range(10):
            est.update(1.0)
        assert est.floor < floor_before * 2  # floor should not have jumped

    def test_is_above_floor(self):
        est = self._make()
        for _ in range(2000):
            est.update(0.001)
        # Something 10× floor should be above
        assert est.is_above_floor(est.floor * 10)
        # Something equal to floor should not be
        assert not est.is_above_floor(est.floor * 0.5)


# ── compute_mechanical_ratio ──────────────────────────────────────────────

class TestComputeMechanicalRatio:
    def test_low_frequency_knock_has_high_ratio(self):
        sr = 44100
        t = np.linspace(0, 512 / sr, 512, endpoint=False)
        # 80 Hz mechanical shock
        knock = (np.sin(2 * np.pi * 80 * t) * np.exp(-t / 0.01)).astype(np.float32)
        ratio = compute_mechanical_ratio(knock, sr)
        assert ratio > 1.0, f"Expected mechanical knock ratio > 1.0, got {ratio}"

    def test_high_frequency_speaker_audio_has_low_ratio(self):
        sr = 44100
        t = np.linspace(0, 512 / sr, 512, endpoint=False)
        # 1500 Hz + 3000 Hz vocal/speaker tone
        audio = (np.sin(2 * np.pi * 1500 * t) + np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        ratio = compute_mechanical_ratio(audio, sr)
        assert ratio < 0.4, f"Expected speaker audio ratio < 0.4, got {ratio}"


# ── compute_spectral_flatness ─────────────────────────────────────────────

class TestComputeSpectralFlatness:
    def test_impulse_has_high_flatness(self):
        # Dirac impulse has flat spectrum -> high flatness
        impulse = np.zeros(512, dtype=np.float32)
        impulse[10] = 1.0
        impulse += np.random.randn(512).astype(np.float32) * 0.05
        flatness = compute_spectral_flatness(impulse)
        assert flatness > 0.4, f"Impulse flatness should be > 0.4, got {flatness}"

    def test_pure_tone_has_low_flatness(self):
        # Pure sinusoidal tone has narrow spectral peak -> very low power flatness (< 0.01)
        t = np.arange(512) / 44100.0
        tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        flatness = compute_spectral_flatness(tone)
        assert flatness < 0.05, f"Pure tone flatness should be < 0.05, got {flatness}"


# ── compute_crest_factor ──────────────────────────────────────────────────

class TestComputeCrestFactor:
    def test_impulse_crest_factor_is_high(self):
        impulse = np.zeros(512, dtype=np.float32)
        impulse[20] = 1.0
        crest = compute_crest_factor(impulse)
        assert crest > 5.0, f"Impulse crest factor should be > 5.0, got {crest}"

    def test_sine_wave_crest_factor_is_low(self):
        t = np.arange(512) / 44100.0
        sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        crest = compute_crest_factor(sine)
        assert crest < 2.0, f"Sine crest factor should be around sqrt(2) (~1.41), got {crest}"

