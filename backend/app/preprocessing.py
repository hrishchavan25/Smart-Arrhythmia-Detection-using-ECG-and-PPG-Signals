import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# STEP 1: Load filtered dataset
# --------------------------------------------------

file_path = "bidmc01_ecg_ppg_filtered.csv"

df = pd.read_csv(file_path)

print("Filtered dataset loaded!")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# STEP 2: Select filtered ECG and PPG
# --------------------------------------------------

signals = df[[
    "ECG_Filtered",
    "PPG_Filtered"
]]

# --------------------------------------------------
# STEP 3: Z-score normalization
# --------------------------------------------------

scaler = StandardScaler()

normalized_signals = scaler.fit_transform(signals)

df["ECG_Normalized"] = normalized_signals[:, 0]
df["PPG_Normalized"] = normalized_signals[:, 1]

# --------------------------------------------------
# STEP 4: Check normalized statistics
# --------------------------------------------------

print("\nNormalized signal statistics:")

print(
    df[[
        "ECG_Normalized",
        "PPG_Normalized"
    ]].describe()
)

# --------------------------------------------------
# STEP 5: Display first 5 samples
# --------------------------------------------------

print("\nFirst 5 normalized samples:")

print(
    df[[
        "ECG_Filtered",
        "ECG_Normalized",
        "PPG_Filtered",
        "PPG_Normalized"
    ]].head()
)

# --------------------------------------------------
# STEP 6: Save normalized dataset
# --------------------------------------------------

output_file = "bidmc01_ecg_ppg_normalized.csv"

df.to_csv(output_file, index=False)

print("\nNormalization completed successfully!")
print("Normalized dataset saved as:", output_file)