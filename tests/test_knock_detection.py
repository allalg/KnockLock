"""
tests/test_knock_detection.py
------------------------------
Unit tests for the KnockDetector state machine.

We synthesise fake audio chunks (noise + synthetic knock impulses)
so these tests run without a real microphone.
"""

import numpy as np
import pytest

from knocklock.config import KnockLockConfig
from knocklock.knock_detection import KnockDetector, KnockEvent


SR = 44100
CHUNK = 512


def _silence(n_chunks: int = 1) -> list[np.ndarray]:
    return [np.zeros(CHUNK, dtype=np.float32)] * n_chunks


def _knock_impulse(amplitude: float = 0.5, duration_chunks: int = 4) -> list[np.ndarray]:
    """Synthesise a short burst at a given amplitude."""
    chunks = []
    for i in range(duration_chunks):
        # Decaying shape: loud at start, trails off
        fade = np.linspace(1.0, 0.1, CHUNK, dtype=np.float32)
        chunks.append(fade * amplitude)
    return chunks


def _run_detector(chunks_with_timestamps, config=None) -> list[KnockEvent]:
    if config is None:
        config = KnockLockConfig()
        # Loosen thresholds so synthetic signals are easy to trigger
        config.detection.amplitude_threshold = 0.05
        config.detection.rise_ratio_threshold = 1.5
        config.detection.min_duration_s = 0.001
        config.detection.max_duration_s = 0.500
        config.detection.noise_floor_alpha = 0.5   # fast floor convergence for tests

    events = []
    det = KnockDetector(config, on_knock=events.append)
    for chunk, ts in chunks_with_timestamps:
        det.process(chunk, ts)
    return events


def _build_stream(pattern: list) -> list[tuple[np.ndarray, float]]:
    """
    Build a list of (chunk, timestamp) from a pattern like:
      [('silence', 20), ('knock', 0.5, 3), ('silence', 30)]
    """
    stream = []
    t = 0.0
    dt = CHUNK / SR
    for item in pattern:
        if item[0] == 'silence':
            for _ in range(item[1]):
                stream.append((np.zeros(CHUNK, dtype=np.float32), t))
                t += dt
        elif item[0] == 'knock':
            amp, n = item[1], item[2]
            for i in range(n):
                fade = np.linspace(1.0, 0.1, CHUNK, dtype=np.float32) * amp
                stream.append((fade, t))
                t += dt
    return stream


class TestKnockDetectorBasic:
    def test_silence_no_knocks(self):
        """Pure silence should produce zero knock events."""
        stream = _build_stream([('silence', 200)])
        events = _run_detector(stream)
        assert len(events) == 0

    def test_single_knock_detected(self):
        """One knock impulse should produce exactly one event."""
        stream = _build_stream([
            ('silence', 50),
            ('knock', 0.6, 5),
            ('silence', 50),
        ])
        events = _run_detector(stream)
        assert len(events) == 1

    def test_multiple_knocks_detected(self):
        """Three separated knocks should produce three events."""
        stream = _build_stream([
            ('silence', 30),
            ('knock', 0.6, 4),
            ('silence', 30),
            ('knock', 0.6, 4),
            ('silence', 30),
            ('knock', 0.6, 4),
            ('silence', 30),
        ])
        events = _run_detector(stream)
        assert len(events) == 3

    def test_event_fields_populated(self):
        stream = _build_stream([
            ('silence', 50),
            ('knock', 0.8, 5),
            ('silence', 50),
        ])
        events = _run_detector(stream)
        assert len(events) >= 1
        e = events[0]
        assert e.index == 1
        assert e.amplitude > 0
        assert e.rms > 0
        assert e.duration_ms > 0
        assert e.timestamp >= 0

    def test_sequential_index(self):
        """KnockEvent index should be monotonically increasing from 1."""
        stream = _build_stream([
            ('silence', 20),
            ('knock', 0.6, 4),
            ('silence', 30),
            ('knock', 0.6, 4),
            ('silence', 20),
        ])
        events = _run_detector(stream)
        assert [e.index for e in events] == list(range(1, len(events) + 1))

    def test_event_str_formatting(self):
        """KnockEvent.__str__ should include key fields."""
        stream = _build_stream([
            ('silence', 50),
            ('knock', 0.5, 5),
            ('silence', 50),
        ])
        events = _run_detector(stream)
        if events:
            s = str(events[0])
            assert "Knock" in s
            assert "t=" in s
            assert "amp=" in s


class TestKnockDetectorEdgeCases:
    def test_very_short_transient_ignored(self):
        """A single-chunk spike below min_duration should be ignored."""
        config = KnockLockConfig()
        config.detection.amplitude_threshold = 0.05
        config.detection.rise_ratio_threshold = 1.5
        config.detection.min_duration_s = 0.050   # 50 ms — longer than one chunk
        config.detection.max_duration_s = 0.500
        config.detection.noise_floor_alpha = 0.5
        # 1-chunk knock = ~11.6 ms < 50 ms min
        stream = _build_stream([
            ('silence', 50),
            ('knock', 0.8, 1),   # only 1 chunk ≈ 11.6 ms
            ('silence', 50),
        ])
        events = _run_detector(stream, config=config)
        assert len(events) == 0

    def test_chunk_count_increments(self):
        stream = _build_stream([('silence', 10)])
        config = KnockLockConfig()
        det = KnockDetector(config)
        for chunk, ts in stream:
            det.process(chunk, ts)
        assert det.chunk_count == 10
