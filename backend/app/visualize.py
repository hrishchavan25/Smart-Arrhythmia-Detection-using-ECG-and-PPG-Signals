import pandas as pd
import matplotlib.pyplot as plt

# Load normalized dataset
file_path = "bidmc01_ecg_ppg_normalized.csv"

df = pd.read_csv(file_path)

# Sampling frequency
fs = 125

# Create time axis
time = df.index / fs

# --------------------------------------------------
# ECG Signal
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(time, df["ECG_Normalized"])

plt.title("Normalized ECG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()

# --------------------------------------------------
# PPG Signal
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(time, df["PPG_Normalized"])

plt.title("Normalized PPG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()