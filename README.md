# KnockLock — Phase 1: Knock Detector

> **Phase 1 of a multi-phase project.**  
> This phase builds only the real-time knock detector.  
> No Windows authentication, no password logic, no pattern recognition yet.

---

## What it does

KnockLock Phase 1 listens to your laptop microphone and detects physical knock sounds (tapping the lid, desk, or palm rest). For every detected knock it prints:

```
Knock #  1  t=  2.431s  amp=0.621  rms=0.1823  dur= 74.5ms  floor=0.00012
```

A live ASCII VU-meter shows the audio level and noise floor in real time.

---

## Project structure

```
knock lock/
├── knocklock/
│   ├── __init__.py          # public package surface
│   ├── config.py            # all configurable thresholds (no hard-coding)
│   ├── audio_capture.py     # microphone input via sounddevice
│   ├── signal_processing.py # high-pass filter, RMS, noise floor estimator
│   ├── knock_detection.py   # state-machine knock detector + KnockEvent
│   └── visualization.py     # real-time ASCII VU-meter
├── tests/
│   ├── test_signal_processing.py
│   └── test_knock_detection.py
├── run_detector.py          # main entry point
├── requirements.txt
└── README.md
```

---

## Installation

### 1 — Prerequisites

- Python 3.10 or newer
- A working microphone (built-in laptop mic is fine)

### 2 — Create a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

> **Windows note:** `sounddevice` uses PortAudio.  
> On most Windows systems this works out of the box via the included binary wheel.  
> If you see a PortAudio error, install it manually:
> ```
> pip install sounddevice[portaudio]
> ```

---

## Running the detector

```powershell
python run_detector.py
```

On launch, it samples 0.4 seconds of ambient silence to calibrate against your room noise:

```text
Calibrating room background noise (0.4s) ... done.
Ambient noise peak: 0.0018 → Set threshold: 0.0120
KnockLock Phase 1 — Knock Detector
Microphone : Microphone Array (Intel® Smart Sound Technology)
Sample rate: 44100 Hz
Threshold  : 0.0120
Refractory : 120 ms
Visualization: ON

Listening … tap your laptop lid, palm rest, or desk. Press Ctrl+C to stop.

[██████████      |       ] peak=0.0084 th=0.012   #  0 knocks

  ► Knock #  1  t=  1.243s  amp=0.038  rms=0.0051  dur= 23.2ms  floor=0.00180
  ► Knock #  2  t=  1.789s  amp=0.045  rms=0.0062  dur= 34.8ms  floor=0.00180
```

Press **Ctrl+C** to stop.

---

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold 0.02` | *auto* | Peak amplitude gate (0.0–1.0). If omitted, automatically calibrated to room ambient level. |
| `--device 9` | *default* | Audio device index (use `--list-devices` to find yours, e.g., WASAPI). |
| `--rise-ratio 2.0` | `2.0` | How sharp the onset must be relative to baseline. |
| `--refractory 0.12` | `0.12` | Minimum gap between knocks (seconds). |
| `--no-viz` | off | Disable the live ASCII meter (print events only). |
| `--list-devices` | — | List all available microphone devices and exit. |

### Examples

```powershell
# Stricter — only very loud, sharp knocks
python run_detector.py --threshold 0.04 --rise-ratio 5.0

# Looser — catch quieter taps
python run_detector.py --threshold 0.01 --rise-ratio 2.0

# No meter, just event log
python run_detector.py --no-viz

# Check available microphones
python run_detector.py --list-devices
```

---

## Running the tests

```powershell
pip install pytest
pytest tests/ -v
```

Tests use synthetic audio — **no microphone required**. They cover:
- RMS and peak calculation
- High-pass filter frequency response and stateful continuity
- Noise-floor adaptive estimator
- Knock detector state machine (silence → no events, impulse → event, short spike → ignored, sequential indexing)

---

## How detection works

```
Microphone
    │
    ▼
High-pass filter (80 Hz)       ← removes hum, desk rumble
    │
    ▼
Short-time RMS energy
    │
    ├──► Adaptive noise floor   ← quiet chunks update background estimate
    │
    ▼
Gate check
  • RMS > amplitude_threshold
  • RMS > noise_floor × snr_multiplier    ← relative to background
  • RMS / prev_RMS > rise_ratio_threshold  ← onset must be a fast transient
    │
    ▼
Duration tracking
  • Stay above threshold for [min, max] window
  • Fall below → emit KnockEvent
    │
    ▼
Refractory period (120 ms)     ← prevents double-counting one knock
    │
    ▼
KnockEvent { index, timestamp, amplitude, rms, duration_ms, noise_floor }
```

---

## Tuning guide

| Symptom | Fix |
|---------|-----|
| Background noise triggers false knocks | Raise `--threshold` or `--rise-ratio` |
| Real knocks are missed | Lower `--threshold` |
| One knock fires twice | Raise `--refractory` |
| Knocks in rapid succession are merged | Lower `--refractory` |
| Long desk rumble triggers knocks | Lower `max_duration_s` in `config.py` |

---

## What's next (Phase 2)

Phase 2 will add **knock pattern extraction** — grouping sequences of knock events, measuring inter-knock timing, and representing them as a pattern vector ready for enrollment and matching.
