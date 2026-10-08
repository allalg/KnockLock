"""
knocklock/__init__.py
---------------------
Public surface of the knocklock package.
"""

from knocklock.config import KnockLockConfig, DEFAULT_CONFIG
from knocklock.audio_capture import AudioCapture, MicrophoneError
from knocklock.signal_processing import HighPassFilter, NoiseFloorEstimator, compute_rms
from knocklock.knock_detection import KnockDetector, KnockEvent
from knocklock.visualization import LiveMeter

__all__ = [
    "KnockLockConfig",
    "DEFAULT_CONFIG",
    "AudioCapture",
    "MicrophoneError",
    "HighPassFilter",
    "NoiseFloorEstimator",
    "compute_rms",
    "KnockDetector",
    "KnockEvent",
    "LiveMeter",
]
