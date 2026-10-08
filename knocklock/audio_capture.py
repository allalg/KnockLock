"""
knocklock/audio_capture.py
--------------------------
Microphone capture using sounddevice.

Provides an AudioCapture context-manager that pushes raw audio chunks
into a thread-safe queue consumed by the processing pipeline.
"""

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from knocklock.config import AudioConfig


class MicrophoneError(RuntimeError):
    """Raised when the microphone cannot be opened or fails mid-stream."""


class AudioCapture:
    """
    Captures microphone audio in real time.

    Usage
    -----
    capture = AudioCapture(config, on_chunk=my_callback)
    with capture:
        time.sleep(10)          # listen for 10 seconds
    """

    def __init__(
        self,
        config: AudioConfig,
        on_chunk: Optional[Callable[[np.ndarray, float], None]] = None,
    ):
        """
        Parameters
        ----------
        config   : AudioConfig
        on_chunk : optional callback(chunk: np.ndarray, timestamp: float)
                   called on the sounddevice audio thread; keep it fast.
                   If None, chunks are pushed to self.queue instead.
        """
        self.config = config
        self._on_chunk = on_chunk
        self.queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=256)
        self._stream: Optional[sd.InputStream] = None
        self._start_time: float = 0.0
        self._frames_captured: int = 0
        self._error: Optional[Exception] = None

    # ── Internal ──────────────────────────────────────────────────────────

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,         # noqa: ANN001 – sounddevice CData struct
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice audio callback — runs on a separate OS thread."""
        if status:
            # Log overflow/underflow but do not crash
            import sys
            print(f"[audio] sounddevice status: {status}", file=sys.stderr)

        if indata.ndim > 1 and indata.shape[1] > 1:
            chunk = np.mean(indata, axis=1)  # average across all microphone capsules
        else:
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()

        elapsed = self._frames_captured / self.config.sample_rate
        timestamp = elapsed  # relative timestamp in seconds
        self._frames_captured += frames

        if self._on_chunk is not None:
            self._on_chunk(chunk, timestamp)
        else:
            try:
                self.queue.put_nowait((chunk, timestamp))
            except queue.Full:
                pass  # drop the oldest chunk rather than block the callback

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream."""
        try:
            self._start_time = time.time()
            self._frames_captured = 0
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                dtype=self.config.dtype,
                device=self.config.device,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise MicrophoneError(
                f"Could not open microphone: {exc}\n"
                "Make sure a microphone is connected and not in use by another app."
            ) from exc

    def stop(self) -> None:
        """Close the microphone stream gracefully."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    # ── Context-manager support ───────────────────────────────────────────

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Info ──────────────────────────────────────────────────────────────

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Return a list of available input devices for diagnostics."""
        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append({"index": idx, "name": dev["name"]})
        return devices

    @staticmethod
    def default_input_device() -> Optional[str]:
        """Return the name of the system default input device, or None."""
        try:
            dev = sd.query_devices(kind="input")
            return dev["name"]
        except Exception:
            return None
