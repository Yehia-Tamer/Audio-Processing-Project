# Audio Processing Project — Clip Matching with the Fourier Transform

A Python tool that locates a short audio clip within a longer recording by comparing their frequency content. Given a full audio file and a query clip, it slides the clip across the recording, computes the Fourier Transform of each segment, and uses cosine similarity to find where the clip occurs — a practical, "Shazam-style" application of signal processing.

The project demonstrates how a theoretical concept like the Fourier Transform can be applied to a real-world problem: analyzing, comparing, and matching signals by their frequency content.

## What It Does

1. **Loads and preprocesses** a WAV file — converts stereo to mono and normalizes different bit depths (16-bit, 32-bit, 8-bit) to a common float range.
2. **Visualizes the signal in the time domain** — plots amplitude over time.
3. **Extracts a query clip** — a short segment to search for within the full recording.
4. **Computes the frequency spectrum** — a one-sided FFT reveals the clip's frequency content.
5. **Matches with a sliding window** — moves the clip across the full audio, taking the FFT of each window and scoring it against the query with cosine similarity.
6. **Detects the best match** — reports the position with the highest similarity score.
7. **Compares and evaluates** — overlays the original clip against the detected segment in both time and frequency domains, and reports correlation and frequency-similarity metrics.

## How It Works

The core idea is that two audio segments containing the same sound share a similar **frequency signature**, even if their raw waveforms differ slightly. So instead of comparing raw samples directly, the tool:

- Uses the **Fast Fourier Transform (FFT)** to convert each segment from the time domain to the frequency domain.
- Compares the resulting magnitude spectra using **cosine similarity** (the normalized dot product), which measures how similar two frequency profiles are regardless of overall loudness.
- Scans the full recording window-by-window, building a similarity-vs-time curve whose peak marks the clip's location.

## Tech Stack

- **Python**
- **NumPy** — FFT, vector math, and signal arrays
- **SciPy** — WAV file reading (`scipy.io.wavfile`)
- **Matplotlib** — time-domain, spectrum, and matching-result visualizations

## Key Functions

| Function | Purpose |
|----------|---------|
| `load_and_preprocess` | Reads a WAV file, converts to mono, normalizes amplitude |
| `visualize_time_domain` | Plots the signal's amplitude over time |
| `extract_query_clip` | Cuts a short clip from the audio at a given start time and duration |
| `compute_one_sided_fft` | Computes the one-sided FFT magnitude and frequency bins |
| `visualize_spectrum` | Plots the frequency spectrum of a signal |
| `sliding_window_matching` | Slides the query across the audio, scoring each window by cosine similarity |
| `detect_best_match` | Finds the window with the highest similarity score |
| `visualize_matching_results` | Plots similarity vs. time, marking the actual and detected positions |
| `compare_signals` | Compares the query and detected segment in time and frequency domains |

## Getting Started

### Prerequisites

- Python 3.8+
- A `.wav` audio file to analyze

### Installation

```bash
pip install numpy scipy matplotlib
```

### Usage

Open the script and set the audio file path in the final line:

```python
main("path/to/your/audio.wav")
```

You can also adjust the matching parameters inside `main`:

- `start_time` — where the query clip is taken from (seconds)
- `clip_duration` — how long the query clip is (seconds)
- `step_size` — how far the window slides each step (seconds); smaller is more precise but slower

Then run:

```bash
python "Signals Project.py"
```

The script prints the detected position and similarity score, and opens plots showing the signal, the query spectrum, the matching curve, and a side-by-side comparison of the original and detected segments.

## About

This project was built to apply Fourier analysis to a concrete signal-processing task — locating an audio clip within a larger recording by its frequency content. It combines FFT-based feature extraction, similarity scoring, and clear visualization to make the underlying signal processing intuitive and verifiable.
