import pandas as pd
import numpy as np

# Load normalized dataset
file_path = "bidmc01_ecg_ppg_normalized.csv"

df = pd.read_csv(file_path)

# Sampling frequency
fs = 125

# Segment duration
segment_duration = 5  # seconds

# Samples per segment
samples_per_segment = fs * segment_duration

print("Samples per segment:", samples_per_segment)

# --------------------------------------------------
# Create segments
# --------------------------------------------------

segments = []

total_samples = len(df)

for start in range(
    0,
    total_samples - samples_per_segment + 1,
    samples_per_segment
):

    end = start + samples_per_segment

    ecg_segment = df["ECG_Normalized"].iloc[start:end].values
    ppg_segment = df["PPG_Normalized"].iloc[start:end].values

    segments.append({
        "ECG": ecg_segment,
        "PPG": ppg_segment
    })

print("Total complete segments:", len(segments))

# --------------------------------------------------
# Check first segment
# --------------------------------------------------

print("\nFirst segment:")
print("ECG samples:", len(segments[0]["ECG"]))
print("PPG samples:", len(segments[0]["PPG"]))

print("\nFirst 5 ECG values:")
print(segments[0]["ECG"][:5])

print("\nFirst 5 PPG values:")
print(segments[0]["PPG"][:5])