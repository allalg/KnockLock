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
        config.detection.min_spectral_flatness = 0.01  # allow synthetic linear ramps
        config.detection.min_crest_factor = 1.4

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
        elif item[0] == 'tone':
            freq, amp, n = item[1], item[2], item[3]
            for i in range(n):
                t_arr = np.linspace(t, t + dt, CHUNK, endpoint=False, dtype=np.float32)
                chunk = (np.sin(2 * np.pi * freq * t_arr) * amp).astype(np.float32)
                stream.append((chunk, t))
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
        """A transient below min_duration should be ignored."""
        config = KnockLockConfig()
        config.detection.amplitude_threshold = 0.05
        config.detection.rise_ratio_threshold = 1.5
        config.detection.min_duration_s = 0.080   # 80 ms min duration
        config.detection.max_duration_s = 0.500
        config.detection.noise_floor_alpha = 0.5
        # 1-chunk knock = ~11.6 ms (< 80 ms min)
        stream = _build_stream([
            ('silence', 50),
            ('knock', 0.8, 1),
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

    def test_sustained_sound_rejected(self):
        """A continuous acoustic sound (speech, speaker audio > max_duration or low mech ratio) must NOT trigger a knock."""
        config = KnockLockConfig()
        config.detection.amplitude_threshold = 0.05
        config.detection.rise_ratio_threshold = 1.5
        config.detection.min_duration_s = 0.005
        config.detection.max_duration_s = 0.100  # 100 ms limit
        # 25 chunks of 1500 Hz vocal/speaker tone = ~290 ms
        stream = _build_stream([
            ('silence', 20),
            ('tone', 1500, 0.4, 25),
            ('silence', 20),
        ])
        events = _run_detector(stream, config=config)
        assert len(events) == 0, f"Expected 0 knocks for sustained speaker audio, got {len(events)}"


class TestKnockDetectorVideoRejection:
    """Tests ensuring laptop speaker video playback is rejected while knocks are detected."""

    def test_video_speech_and_music_rejected(self):
        """Video dialogue (140-280 Hz harmonics with pauses) must produce 0 false knocks."""
        config = KnockLockConfig()
        config.detection.amplitude_threshold = 0.035
        events = []
        det = KnockDetector(config, on_knock=events.append)
        dt = CHUNK / SR
        t = 0.0

        # Initial quiet baseline
        for _ in range(30):
            det.process(np.zeros(CHUNK, dtype=np.float32), t)
            t += dt

        # Simulate video playing for 100 chunks (speech dialogue + pauses)
        for i in range(100):
            t_chunk = np.arange(CHUNK) / SR
            if i % 15 < 10:
                # Spoken vowel
                vocal = (0.15 * np.sin(2 * np.pi * 140 * t_chunk) + 0.10 * np.sin(2 * np.pi * 280 * t_chunk)).astype(np.float32)
                chunk = vocal + np.random.randn(CHUNK).astype(np.float32) * 0.005
            else:
                # Inter-word pause
                chunk = np.random.randn(CHUNK).astype(np.float32) * 0.002
            det.process(chunk, t)
            t += dt

        assert len(events) == 0, f"Expected 0 knocks from video playback, got {len(events)}"

    def test_knock_detected_during_video(self):
        """A physical knock (table thump/tap) occurring while video is playing should be detected."""
        config = KnockLockConfig()
        config.detection.amplitude_threshold = 0.035
        events = []
        det = KnockDetector(config, on_knock=events.append)
        dt = CHUNK / SR
        t = 0.0

        # 1. Video playback (50 chunks with natural vocal rhythm)
        for i in range(50):
            t_chunk = np.arange(CHUNK) / SR
            vocal = (0.08 * np.sin(2 * np.pi * 160 * t_chunk)).astype(np.float32)
            chunk = vocal if (i % 15 < 10) else np.random.randn(CHUNK).astype(np.float32) * 0.005
            det.process(chunk, t)
            t += dt

        # 2. Knock impulse (sharp impact + 35ms decay)
        t_impulse = np.arange(CHUNK) / SR
        knock_chunk = np.zeros(CHUNK, dtype=np.float32)
        knock_chunk[10:30] = np.random.randn(20).astype(np.float32) * 0.7
        knock_chunk[30:] = (0.2 * np.exp(-t_impulse[30:] / 0.01) * np.sin(2 * np.pi * 80 * t_impulse[30:])).astype(np.float32)
        det.process(knock_chunk, t)
        t += dt

        # Tail decay
        det.process(knock_chunk * 0.15, t)
        t += dt
        det.process(np.random.randn(CHUNK).astype(np.float32) * 0.005, t)
        t += dt

        # 3. Resume video playback (50 chunks)
        for i in range(50):
            t_chunk = np.arange(CHUNK) / SR
            vocal = (0.08 * np.sin(2 * np.pi * 160 * t_chunk)).astype(np.float32)
            chunk = vocal if (i % 15 < 10) else np.random.randn(CHUNK).astype(np.float32) * 0.005
            det.process(chunk, t)
            t += dt

        assert len(events) == 1, f"Expected exactly 1 knock during video playback, got {len(events)}"

