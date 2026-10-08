"""
knocklock/visualization.py
---------------------------
Real-time ASCII debug display for Phase 1.

Shows:
  • A live audio-level meter
  • Noise floor marker
  • The amplitude threshold marker
  • Highlighted knock events (flash of "KNOCK!" text)
"""

import sys
import threading
import time

from knocklock.config import VisualizationConfig, DetectionConfig


_RESET = "\033[0m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"


class LiveMeter:
    """
    Prints an updating ASCII VU-meter to the terminal.

    Thread-safe: update() can be called from the audio thread;
    render() should be called from the main thread.
    """

    def __init__(
        self,
        vis_config: VisualizationConfig,
        det_config: DetectionConfig,
    ):
        self._bar_width = vis_config.bar_width
        self._threshold = det_config.amplitude_threshold
        self._refresh_n = vis_config.refresh_every_n_chunks

        self._lock = threading.Lock()
        self._current_rms: float = 0.0
        self._current_peak: float = 0.0
        self._noise_floor: float = 0.0
        self._knock_flash: bool = False
        self._knock_flash_until: float = 0.0
        self._knock_count: int = 0
        self._chunk_calls: int = 0

    def update(
        self,
        rms: float,
        peak: float,
        noise_floor: float,
        knock_count: int,
    ) -> None:
        """Called after each processed chunk; stores state for next render."""
        with self._lock:
            self._current_rms = rms
            self._current_peak = peak
            self._noise_floor = noise_floor
            self._knock_count = knock_count
            self._chunk_calls += 1

    def flash_knock(self) -> None:
        """Trigger the KNOCK! flash indicator for ~0.3 s."""
        with self._lock:
            self._knock_flash = True
            self._knock_flash_until = time.monotonic() + 0.30

    def should_render(self) -> bool:
        with self._lock:
            return self._chunk_calls % self._refresh_n == 0

    def render(self) -> None:
        """Print one line to stdout (overwrites the previous line)."""
        with self._lock:
            rms = self._current_rms
            peak = self._current_peak
            floor = self._noise_floor
            count = self._knock_count
            flash = self._knock_flash and time.monotonic() < self._knock_flash_until
            if self._knock_flash and not flash:
                self._knock_flash = False

        # ── Build the scaled bar ──────────────────────────────────────────
        # Scale so threshold is at 45% of the bar width
        w = min(self._bar_width, 26)  # keep under 26 chars to prevent wrapping
        scale = (w * 0.45) / max(self._threshold, 1e-4)

        filled = int(min(max(peak * scale, 0.0), w))
        threshold_pos = int(min(max(self._threshold * scale, 0.0), w - 1))
        floor_pos = int(min(max(floor * scale, 0.0), w - 1))

        bar_chars = []
        for i in range(w):
            if i < filled:
                if i >= threshold_pos:
                    bar_chars.append(_RED + "█" + _RESET)
                else:
                    bar_chars.append(_GREEN + "█" + _RESET)
            elif i == threshold_pos:
                bar_chars.append(_YELLOW + "|" + _RESET)
            elif i == floor_pos:
                bar_chars.append(_CYAN + "·" + _RESET)
            else:
                bar_chars.append(" ")

        bar = "".join(bar_chars)

        # ── Knock indicator ───────────────────────────────────────────
        if flash:
            knock_str = f" {_BOLD}{_RED}★ KNOCK!{_RESET} #{count}"
        else:
            knock_str = f"   #{count:>2} knocks"

        # Keep formatted line within ~75 columns to prevent PowerShell line-wrapping
        line = (
            f"\r[{bar}] "
            f"peak={peak:.4f} "
            f"th={self._threshold:.3f}"
            f"{knock_str}  "
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def print_knock(self, event) -> None:
        """
        Print a knock event on a fresh line above the live meter.
        The meter will keep updating below.
        """
        # Move to a fresh line, print event, then the meter continues
        sys.stdout.write(f"\n{_BOLD}{_RED}  ► {event}{_RESET}\n")
        sys.stdout.flush()
