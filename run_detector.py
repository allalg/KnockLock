"""
run_detector.py
---------------
Main entry point for KnockLock Phase 1.

Run:
    python run_detector.py              # default config, debug meter on
    python run_detector.py --no-viz     # plain text only
    python run_detector.py --threshold 0.03 --refractory 0.15

Press Ctrl+C to stop.
"""

import argparse
import queue
import signal
import sys
import time

from knocklock.audio_capture import AudioCapture, MicrophoneError
from knocklock.config import KnockLockConfig
from knocklock.knock_detection import KnockDetector, KnockEvent
from knocklock.signal_processing import compute_rms, compute_peak
from knocklock.visualization import LiveMeter


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KnockLock Phase 1 — real-time knock detector"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Amplitude threshold (0.0–1.0). If omitted, automatically calibrated to room noise.",
    )
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (use --list-devices to view options)",
    )
    p.add_argument(
        "--rise-ratio",
        type=float,
        default=None,
        help="Rise-ratio needed to trigger knock onset (default: 2.0)",
    )
    p.add_argument(
        "--refractory",
        type=float,
        default=None,
        help="Minimum gap between knocks in seconds (default: 0.12)",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable the live ASCII meter (print knock events only)",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input devices and exit",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device listing ────────────────────────────────────────────────
    if args.list_devices:
        devices = AudioCapture.list_input_devices()
        print("Available input devices:")
        for d in devices:
            print(f"  [{d['index']:>2}] {d['name']}")
        default = AudioCapture.default_input_device()
        print(f"\nDefault: {default}")
        sys.exit(0)

    # ── Build config ──────────────────────────────────────────────────
    config = KnockLockConfig()

    if args.device is not None:
        config.audio.device = args.device
    if args.rise_ratio is not None:
        config.detection.rise_ratio_threshold = args.rise_ratio
    if args.refractory is not None:
        config.detection.refractory_s = args.refractory
    if args.no_viz:
        config.visualization.enabled = False

    # ── Auto-calibration if no manual threshold provided ─────────────
    if args.threshold is not None:
        config.detection.amplitude_threshold = args.threshold
        print(f"Using manual threshold: {config.detection.amplitude_threshold:.4f}")
    else:
        print("Calibrating room background noise (0.4s) ...", end="", flush=True)
        calibration_peaks = []

        def _calib_chunk(chunk, _):
            calibration_peaks.append(float(compute_peak(chunk)))

        try:
            with AudioCapture(config.audio, on_chunk=_calib_chunk):
                time.sleep(0.4)
            ambient_peak = max(calibration_peaks) if calibration_peaks else 0.003
            # Set threshold comfortably above ambient noise
            calibrated_th = max(ambient_peak * 3.0, 0.012)
            config.detection.amplitude_threshold = calibrated_th
            print(f" done.")
            print(f"Ambient noise peak: {ambient_peak:.4f} → Set threshold: {calibrated_th:.4f}")
        except Exception as e:
            print(f" (fallback to default 0.015: {e})")
            config.detection.amplitude_threshold = 0.015

    # ── Set up visualization ──────────────────────────────────────────
    meter = LiveMeter(config.visualization, config.detection) if config.visualization.enabled else None

    # ── Knock callback ────────────────────────────────────────────────
    def on_knock(event: KnockEvent) -> None:
        if meter:
            meter.flash_knock()
            meter.print_knock(event)
        else:
            print(event)
        sys.stdout.flush()

    # ── Detector ──────────────────────────────────────────────────────
    detector = KnockDetector(config, on_knock=on_knock)

    # ── Graceful Ctrl-C ───────────────────────────────────────────────
    stop_event = False

    def _sigint_handler(sig, frame):
        nonlocal stop_event
        stop_event = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Startup Info ──────────────────────────────────────────────────
    default_dev = AudioCapture.default_input_device()
    print(f"\nKnockLock Phase 1 — Knock Detector")
    print(f"Microphone : {default_dev or '(unknown)'}")
    print(f"Sample rate: {config.audio.sample_rate} Hz")
    print(f"Threshold  : {config.detection.amplitude_threshold:.4f}")
    print(f"Refractory : {config.detection.refractory_s * 1000:.0f} ms")
    print(f"Visualization: {'ON' if meter else 'OFF'}")
    print("\nListening … tap your laptop lid, palm rest, or desk. Press Ctrl+C to stop.\n")

    # ── Run (main thread consumes from queue to protect audio thread) ──
    try:
        with AudioCapture(config.audio) as capture:
            while not stop_event:
                try:
                    chunk, timestamp = capture.queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                # Run knock detection
                detector.process(chunk, timestamp)

                # Update meter on the main thread
                if meter:
                    meter.update(
                        detector._prev_rms,
                        detector._prev_peak,
                        detector.noise_floor,
                        detector.knock_count,
                    )
                    if meter.should_render():
                        meter.render()
    except MicrophoneError as exc:
        print(f"\n\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\nStopped. Total knocks detected: {detector.knock_count}")


if __name__ == "__main__":
    main()
