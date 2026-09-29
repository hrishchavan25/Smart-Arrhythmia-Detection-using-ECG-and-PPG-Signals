import pandas as pd
import numpy as np

# 1. Load the CSV
df = pd.read_csv("cardiobench_sample.csv")

print("Original data:")
print(df.head())

# 2. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Remove rows where ECG or PPG is missing
df = df.dropna(subset=["ecg", "ppg"])

# 4. Convert ECG and PPG to numbers
df["ecg"] = pd.to_numeric(df["ecg"], errors="coerce")
df["ppg"] = pd.to_numeric(df["ppg"], errors="coerce")

# 5. Remove invalid values
df = df.dropna(subset=["ecg", "ppg"])

# 6. Normalize ECG
df["ecg_normalized"] = (
    (df["ecg"] - df["ecg"].mean()) / df["ecg"].std()
)

# 7. Normalize PPG
df["ppg_normalized"] = (
    (df["ppg"] - df["ppg"].mean()) / df["ppg"].std()
)

# 8. Save processed data
df.to_csv("processed_cardiobench.csv", index=False)

print("\nPreprocessing completed!")
print("Processed data saved as: processed_cardiobench.csv")

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns)
import matplotlib.pyplot as plt

# Take first 1000 samples
sample = df.iloc[:1000]

# ECG graph
plt.figure(figsize=(12, 4))
plt.plot(sample["ecg_normalized"])
plt.title("Normalized ECG Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# PPG graph
plt.figure(figsize=(12, 4))
plt.plot(sample["ppg_normalized"])
plt.title("Normalized PPG Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid()
plt.show()