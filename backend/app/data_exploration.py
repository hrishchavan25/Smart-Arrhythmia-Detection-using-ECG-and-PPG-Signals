# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# ECG + PPG DATA EXPLORATION
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
# Get the path of the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bidmc01_ecg_ppg.csv"
)

# Read CSV file
df = pd.read_csv(DATA_PATH, nrows=10000)
print("\nDATASET LOADED SUCCESSFULLY")
print("\nFirst 5 rows:")
print(df.head())
# ---------------------------------------------------------
# 2. DATASET INFORMATION
# ---------------------------------------------------------
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nDataset Information:")
df.info()
# ---------------------------------------------------------
# 3. CHECK MISSING VALUES
# ---------------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())
# ---------------------------------------------------------
# 4. CHECK DUPLICATE ROWS
# ---------------------------------------------------------
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# ---------------------------------------------------------
# 5. STATISTICAL ANALYSIS
# ---------------------------------------------------------
print("\nStatistical Summary:")
print(df.describe())
# ---------------------------------------------------------
# 6. INDIVIDUAL SIGNAL STATISTICS
# ---------------------------------------------------------
signals = ["ECG", "PPG"]
for signal in signals:
    print("\nStatistics for", signal)
    print("Mean:", df[signal].mean())
    print("Standard Deviation:", df[signal].std())
    print("Minimum:", df[signal].min())
    print("Maximum:", df[signal].max())
    print("Median:", df[signal].median())
# ---------------------------------------------------------
# 7. VISUALIZE RAW ECG SIGNAL
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(df["ECG"], color="blue", linewidth=0.8)
plt.title("Raw ECG Signal")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 8. VISUALIZE RAW PPG SIGNAL
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(df["PPG"], color="red", linewidth=0.8)
plt.title("Raw PPG Signal")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 9. COMBINED ECG AND PPG VISUALIZATION
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(df["ECG"], color="blue", linewidth=0.8)
axes[0].set_title("ECG Signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)
axes[1].plot(df["PPG"], color="red", linewidth=0.8)
axes[1].set_title("PPG Signal")
axes[1].set_xlabel("Sample Number")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 10. CORRELATION ANALYSIS
# ---------------------------------------------------------
correlation = df[["ECG", "PPG"]].corr()
print("\nECG-PPG Correlation:")
print(correlation)
plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("ECG and PPG Correlation")
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# END OF DATA EXPLORATION
# ---------------------------------------------------------
print("\nDATA EXPLORATION COMPLETED SUCCESSFULLY!")