
# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# SAMPLE-LEVEL LABEL ANALYSIS
# ---------------------------------------------------------

from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "sukju/cardiologent_dataset",
    "waveforms",
    split="train",
    streaming=True
)


# ---------------------------------------------------------
# 2. INSPECT FIRST 100 WINDOWS
# ---------------------------------------------------------

MAX_SAMPLES = 100

records = []

for i, sample in enumerate(dataset):

    if i >= MAX_SAMPLES:
        break

    mask = np.asarray(
        sample["mask"],
        dtype=int
    )

    records.append({

        "window_id": sample["window_id"],

        "patient_id": sample["caseid"],

        "split": sample["split"],

        "total_samples": len(mask),

        "normal_count": np.sum(mask == 0),

        "pvc_count": np.sum(mask == 4),

        "normal_percentage": (
            np.sum(mask == 0) / len(mask)
        ) * 100,

        "pvc_percentage": (
            np.sum(mask == 4) / len(mask)
        ) * 100,

        "unique_labels": len(np.unique(mask))

    })


# ---------------------------------------------------------
# 3. CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(records)

print("\nLABEL ANALYSIS COMPLETED!")

print("\nDataset shape:", df.shape)


# ---------------------------------------------------------
# 4. DISPLAY WINDOW-LEVEL LABEL DISTRIBUTION
# ---------------------------------------------------------

print("\nFirst 10 windows:")

print(df.head(10).to_string(index=False))


# ---------------------------------------------------------
# 5. IDENTIFY MIXED WINDOWS
# ---------------------------------------------------------

mixed_windows = df[
    (df["normal_count"] > 0)
    & (df["pvc_count"] > 0)
]

normal_only = df[
    (df["normal_count"] > 0)
    & (df["pvc_count"] == 0)
]

pvc_only = df[
    (df["pvc_count"] > 0)
    & (df["normal_count"] == 0)
]

print("\nWINDOW DISTRIBUTION")

print("Normal-only windows:", len(normal_only))

print("PVC-only windows:", len(pvc_only))

print("Mixed windows:", len(mixed_windows))


# ---------------------------------------------------------
# 6. DISPLAY OVERALL SAMPLE DISTRIBUTION
# ---------------------------------------------------------

total_normal = df["normal_count"].sum()

total_pvc = df["pvc_count"].sum()

total_samples = total_normal + total_pvc

print("\nSAMPLE-LEVEL DISTRIBUTION")

print(
    "Normal samples:",
    total_normal
)

print(
    "PVC samples:",
    total_pvc
)

print(
    "Normal percentage:",
    round(total_normal / total_samples * 100, 2)
)

print(
    "PVC percentage:",
    round(total_pvc / total_samples * 100, 2)
)


# ---------------------------------------------------------
# 7. VISUALIZE WINDOW DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    ["Normal Only", "PVC Only", "Mixed"],
    [
        len(normal_only),
        len(pvc_only),
        len(mixed_windows)
    ]
)

plt.title("Window-Level Label Distribution")

plt.xlabel("Window Category")

plt.ylabel("Number of Windows")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 8. VISUALIZE SAMPLE-LEVEL DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    ["Normal", "PVC"],
    [total_normal, total_pvc]
)

plt.title("Sample-Level Label Distribution")

plt.xlabel("Rhythm Class")

plt.ylabel("Number of Samples")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 9. SAVE RESULTS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "label_analysis.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nAnalysis saved successfully!")

print("Output file:", OUTPUT_PATH)

print("\nLABEL ANALYSIS FINISHED!")