import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def load_and_preprocess(file_path):
    fs, audio = wavfile.read(file_path)

    if len(audio.shape) == 2 and audio.shape[1] == 2:
        audio = np.mean(audio, axis=1)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)

    return fs, audio


def visualize_time_domain(fs, audio, duration_seconds=0.01, title="Audio Signal - Time Domain"):
    total_samples = len(audio)
    total_time = total_samples / fs
    samples_to_plot = int(duration_seconds * fs)

    if samples_to_plot > total_samples:
        samples_to_plot = total_samples
        duration_seconds = total_time

    time_axis = np.arange(samples_to_plot) / fs
    signal_portion = audio[0:samples_to_plot]

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, signal_portion, 'b-', linewidth=0.8)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title(f'{title}\nFirst {duration_seconds * 1000:.2f} milliseconds', fontsize=14)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    return time_axis, signal_portion


def extract_query_clip(audio, fs, start_time, duration):
    start_sample = int(start_time * fs)
    num_samples = int(duration * fs)
    query_clip = audio[start_sample:start_sample + num_samples]
    return query_clip


def compute_one_sided_fft(signal, fs):
    N = len(signal)
    fft_complex = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_complex)
    one_sided_magnitude = fft_magnitude[0:N // 2]
    frequencies = np.fft.fftfreq(N, 1 / fs)[0:N // 2]
    return frequencies, one_sided_magnitude


def visualize_spectrum(signal, fs):
    frequencies, magnitude = compute_one_sided_fft(signal, fs)
    plt.figure(figsize=(12, 5))
    plt.plot(frequencies, magnitude, 'r-', linewidth=0.8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum of Query Clip')
    plt.grid(True, alpha=0.5)
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.show()
    return frequencies, magnitude


def sliding_window_matching(full_audio, fs, query_clip, step_size=0.1):
    _, query_mag = compute_one_sided_fft(query_clip, fs)
    query_norm = np.linalg.norm(query_mag)

    query_len = len(query_clip)
    full_len = len(full_audio)
    step_samples = int(step_size * fs)
    num_windows = (full_len - query_len) // step_samples + 1

    scores = np.zeros(num_windows)
    positions = np.zeros(num_windows)

    for i in range(num_windows):
        start_sample = i * step_samples
        window = full_audio[start_sample:start_sample + query_len]
        _, window_mag = compute_one_sided_fft(window, fs)
        dot_product = np.dot(query_mag, window_mag)
        window_norm = np.linalg.norm(window_mag)

        if window_norm > 0 and query_norm > 0:
            score = dot_product / (query_norm * window_norm)
        else:
            score = 0

        scores[i] = score
        positions[i] = start_sample / fs

    return scores, positions


def detect_best_match(scores, positions):
    best_index = np.argmax(scores)
    best_time = positions[best_index]
    best_score = scores[best_index]
    return best_score, best_time, best_index


def visualize_matching_results(scores, positions, actual_time, detected_time, query_duration):
    plt.figure(figsize=(14, 6))
    plt.plot(positions, scores, 'b-', linewidth=0.8, label='Similarity Score', alpha=0.8)

    plt.axvline(x=actual_time, color='g', linestyle='-', linewidth=2.5,
                label=f'Actual Position: {actual_time:.2f}s', alpha=0.8)
    plt.axvspan(actual_time, actual_time + query_duration,
                alpha=0.2, color='green', label='Actual Clip Region')

    plt.axvline(x=detected_time, color='r', linestyle='--', linewidth=2.5,
                label=f'Detected Position: {detected_time:.2f}s', alpha=0.8)

    peak_idx = np.argmax(scores)
    peak_time = positions[peak_idx]
    peak_score = scores[peak_idx]
    plt.plot(peak_time, peak_score, 'ro', markersize=10, label=f'Peak Score: {peak_score:.3f}')

    if actual_time != detected_time:
        diff = abs(detected_time - actual_time)
        y_pos = max(scores) * 0.9
        plt.annotate(f'Error: {diff:.3f}s',
                     xy=((actual_time + detected_time) / 2, y_pos),
                     xytext=((actual_time + detected_time) / 2, y_pos + 0.1),
                     ha='center', fontsize=11, weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Similarity Score (Cosine Similarity)', fontsize=12)
    plt.title('Audio Matching Results: Similarity Score vs. Time', fontsize=14, weight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)

    plt.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='High confidence (0.8)')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Medium confidence (0.5)')

    stats_text = f'Statistics:\nPeak Score: {peak_score:.4f}\nDetection Error: {abs(actual_time - detected_time):.3f}s'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()


def compare_signals(query_clip, detected_segment, fs):
    time_query = np.arange(len(query_clip)) / fs
    time_detected = np.arange(len(detected_segment)) / fs

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_query, query_clip, 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Original Query Clip (Time Domain)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_query[-1])

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time_detected, detected_segment, 'r-', linewidth=0.8)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('Detected Matching Segment (Time Domain)', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_detected[-1])

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time_query, query_clip, 'b-', linewidth=0.8, alpha=0.7, label='Original Clip')
    ax3.plot(time_detected, detected_segment, 'r-', linewidth=0.8, alpha=0.7, label='Detected Segment')
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_title('Overlay Comparison (Time Domain)', fontsize=12, weight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(time_query[-1], time_detected[-1]))

    ax4 = plt.subplot(2, 2, 4)
    freq_q, mag_q = compute_one_sided_fft(query_clip, fs)
    freq_d, mag_d = compute_one_sided_fft(detected_segment, fs)

    limit_hz = 5000
    idx_limit = np.where(freq_q <= limit_hz)[0]

    ax4.plot(freq_q[idx_limit], mag_q[idx_limit], 'b-', linewidth=0.8, alpha=0.7, label='Original Clip')
    ax4.plot(freq_d[idx_limit], mag_d[idx_limit], 'r-', linewidth=0.8, alpha=0.7, label='Detected Segment')
    ax4.set_xlabel('Frequency (Hz)', fontsize=11)
    ax4.set_ylabel('Magnitude', fontsize=11)
    ax4.set_title('Frequency Domain Comparison', fontsize=12, weight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, limit_hz)

    plt.tight_layout()
    plt.show()

    correlation = np.corrcoef(query_clip, detected_segment)[0, 1]
    freq_similarity = np.dot(mag_q, mag_d) / (np.linalg.norm(mag_q) * np.linalg.norm(mag_d))

    print("\n=== Signal Comparison Metrics ===")
    print(f"Time domain correlation: {correlation:.4f} (1.0 = identical)")
    print(f"Frequency domain similarity: {freq_similarity:.4f} (1.0 = identical)")

    if correlation > 0.95:
        print("✅ Signals are nearly identical - Good match!")
    elif correlation > 0.8:
        print("👍 Signals are very similar - Acceptable match")
    elif correlation > 0.6:
        print("⚠️ Signals are moderately similar - Check detection")
    else:
        print("❌ Signals are different - Detection may be incorrect")

    return correlation, freq_similarity


# ================== MAIN ==================
def main(path):
    # ---------- Configuration ----------
    audio_file = path # depends on .wav file location 
    start_time = 2.0
    clip_duration = 1.0
    step_size = 0.1

    # 1. Load audio
    fs, full_audio = load_and_preprocess(audio_file)
    print(f"Sampling frequency: {fs} Hz")
    print(f"Length of full signal: {len(full_audio)} samples ({len(full_audio)/fs:.2f} seconds)")

    # 2. Full signal time domain - use total duration to plot everything
    total_duration = len(full_audio) / fs
    visualize_time_domain(fs, full_audio, duration_seconds=total_duration, title="Full Audio Signal")

    # 3. Extract query clip
    query = extract_query_clip(full_audio, fs, start_time, clip_duration)
    print(f"Clip length: {len(query)} samples ({clip_duration} seconds)")

    # 4. Visualise clip in time domain (full clip duration)
    visualize_time_domain(fs, query, duration_seconds=clip_duration, title="Query Clip")

    # 5. Visualise clip frequency spectrum (no max_freq parameter)
    visualize_spectrum(query, fs)

    # 6 & 7. Matching & detection
    scores, positions = sliding_window_matching(full_audio, fs, query, step_size)
    best_score, detected_time, _ = detect_best_match(scores, positions)

    print(f"\nOriginal clip position: {start_time:.2f} s")
    print(f"Detected position: {detected_time:.2f} s")
    print(f"Best similarity score: {best_score:.4f}")

    # 8. Visualise matching results
    visualize_matching_results(scores, positions, start_time, detected_time, clip_duration)

    # 9. Compare original clip with detected segment
    detected_start = int(detected_time * fs)
    detected_segment = full_audio[detected_start:detected_start + len(query)]
    compare_signals(query, detected_segment, fs)


main("") # <---- enter path here 