
# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# ECG + PPG SIGNAL PREPROCESSING
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bidmc01_ecg_ppg.csv"
)

# Load only the first 10,000 rows
df = pd.read_csv(DATA_PATH, nrows=10000)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. SAMPLING FREQUENCY
# ---------------------------------------------------------

FS = 125

print("\nSampling Frequency:", FS, "Hz")


# ---------------------------------------------------------
# 3. EXTRACT RAW SIGNALS
# ---------------------------------------------------------

ecg_raw = df["ECG"].to_numpy(dtype=float)

ppg_raw = df["PPG"].to_numpy(dtype=float)

print("\nECG samples:", len(ecg_raw))
print("PPG samples:", len(ppg_raw))


# ---------------------------------------------------------
# 4. BANDPASS FILTER FUNCTION
# ---------------------------------------------------------

def apply_bandpass(signal, lowcut, highcut, fs):

    nyquist = fs / 2

    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


# ---------------------------------------------------------
# 5. FILTER ECG SIGNAL
# ---------------------------------------------------------

# ECG frequency range: 0.5 Hz to 40 Hz

ecg_filtered = apply_bandpass(
    ecg_raw,
    0.5,
    40,
    FS
)

print("\nECG filtering completed!")


# ---------------------------------------------------------
# 6. FILTER PPG SIGNAL
# ---------------------------------------------------------

# PPG frequency range: 0.5 Hz to 8 Hz

ppg_filtered = apply_bandpass(
    ppg_raw,
    0.5,
    8,
    FS
)

print("PPG filtering completed!")


# ---------------------------------------------------------
# 7. NORMALIZATION FUNCTION
# ---------------------------------------------------------

def normalize_signal(signal):

    mean = np.mean(signal)

    std = np.std(signal)

    if std == 0:
        return signal - mean

    normalized = (signal - mean) / std

    return normalized


# ---------------------------------------------------------
# 8. NORMALIZE SIGNALS
# ---------------------------------------------------------

ecg_normalized = normalize_signal(ecg_filtered)

ppg_normalized = normalize_signal(ppg_filtered)

print("\nNormalization completed!")


# ---------------------------------------------------------
# 9. CREATE TIME AXIS
# ---------------------------------------------------------

time = np.arange(len(ecg_raw)) / FS


# ---------------------------------------------------------
# 10. VISUALIZE ECG BEFORE AND AFTER FILTERING
# ---------------------------------------------------------

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)

plt.plot(time, ecg_raw, color="blue", linewidth=0.8)

plt.title("Raw ECG Signal")

plt.ylabel("Amplitude")

plt.grid(True)


plt.subplot(2, 1, 2)

plt.plot(time, ecg_filtered, color="green", linewidth=0.8)

plt.title("Filtered ECG Signal")

plt.xlabel("Time (seconds)")

plt.ylabel("Amplitude")

plt.grid(True)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 11. VISUALIZE PPG BEFORE AND AFTER FILTERING
# ---------------------------------------------------------

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)

plt.plot(time, ppg_raw, color="red", linewidth=0.8)

plt.title("Raw PPG Signal")

plt.ylabel("Amplitude")

plt.grid(True)


plt.subplot(2, 1, 2)

plt.plot(time, ppg_filtered, color="green", linewidth=0.8)

plt.title("Filtered PPG Signal")

plt.xlabel("Time (seconds)")

plt.ylabel("Amplitude")

plt.grid(True)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 12. VISUALIZE NORMALIZED SIGNALS
# ---------------------------------------------------------

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)

plt.plot(time, ecg_normalized, color="blue", linewidth=0.8)

plt.title("Normalized ECG Signal")

plt.ylabel("Normalized Amplitude")

plt.grid(True)


plt.subplot(2, 1, 2)

plt.plot(time, ppg_normalized, color="red", linewidth=0.8)

plt.title("Normalized PPG Signal")

plt.xlabel("Time (seconds)")

plt.ylabel("Normalized Amplitude")

plt.grid(True)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 13. SIGNAL SEGMENTATION
# ---------------------------------------------------------

SEGMENT_DURATION = 10

SEGMENT_SIZE = FS * SEGMENT_DURATION

ecg_segments = []

ppg_segments = []

total_samples = len(ecg_normalized)

for start in range(0, total_samples, SEGMENT_SIZE):

    end = start + SEGMENT_SIZE

    if end > total_samples:
        break

    ecg_segment = ecg_normalized[start:end]

    ppg_segment = ppg_normalized[start:end]

    ecg_segments.append(ecg_segment)

    ppg_segments.append(ppg_segment)


ecg_segments = np.array(ecg_segments)

ppg_segments = np.array(ppg_segments)


print("\nSEGMENTATION COMPLETED")

print("Segment duration:", SEGMENT_DURATION, "seconds")

print("Samples per segment:", SEGMENT_SIZE)

print("Number of ECG segments:", len(ecg_segments))

print("Number of PPG segments:", len(ppg_segments))

print("ECG segment shape:", ecg_segments.shape)

print("PPG segment shape:", ppg_segments.shape)


# ---------------------------------------------------------
# 14. VISUALIZE FIRST SEGMENT
# ---------------------------------------------------------

segment_time = np.arange(SEGMENT_SIZE) / FS

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)

plt.plot(
    segment_time,
    ecg_segments[0],
    color="blue"
)

plt.title("First ECG Segment (10 Seconds)")

plt.ylabel("Normalized Amplitude")

plt.grid(True)


plt.subplot(2, 1, 2)

plt.plot(
    segment_time,
    ppg_segments[0],
    color="red"
)

plt.title("First PPG Segment (10 Seconds)")

plt.xlabel("Time (seconds)")

plt.ylabel("Normalized Amplitude")

plt.grid(True)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 15. SAVE PROCESSED DATA
# ---------------------------------------------------------

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_signals.npz"
)

np.savez(
    OUTPUT_PATH,
    ecg_raw=ecg_raw,
    ppg_raw=ppg_raw,
    ecg_filtered=ecg_filtered,
    ppg_filtered=ppg_filtered,
    ecg_normalized=ecg_normalized,
    ppg_normalized=ppg_normalized,
    ecg_segments=ecg_segments,
    ppg_segments=ppg_segments,
    sampling_frequency=FS
)

print("\nProcessed signals saved successfully!")

print("Output file:", OUTPUT_PATH)

print("\nSIGNAL PREPROCESSING COMPLETED!")