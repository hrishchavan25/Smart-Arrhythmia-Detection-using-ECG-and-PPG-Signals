import wfdb
import pandas as pd
import os

# Get record from PhysioNet
record = wfdb.rdrecord("bidmc01", pn_dir="bidmc/1.0.0")

print("Signals:", record.sig_name)
print("Sampling frequency:", record.fs)
print("Number of samples:", record.sig_len)

# Find ECG and PPG
ecg_index = record.sig_name.index("II,")
ppg_index = record.sig_name.index("PLETH,")

# Extract ECG and PPG
ecg = record.p_signal[:, ecg_index]
ppg = record.p_signal[:, ppg_index]

print("ECG extracted successfully!")
print("PPG extracted successfully!")
print("ECG samples:", len(ecg))
print("PPG samples:", len(ppg))

# Create table
data = pd.DataFrame({
    "ECG": ecg,
    "PPG": ppg
})

# Save CSV
data.to_csv("bidmc01_ecg_ppg.csv", index=False)

print("ECG + PPG data saved successfully!")
print("CSV file created: bidmc01_ecg_ppg.csv")

# Check exactly where the CSV was saved
print("CSV exists:", os.path.exists("bidmc01_ecg_ppg.csv"))
print("CSV full path:", os.path.abspath("bidmc01_ecg_ppg.csv"))
