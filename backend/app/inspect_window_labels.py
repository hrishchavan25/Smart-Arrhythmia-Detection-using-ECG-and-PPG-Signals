
import os
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Set paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "multi_patient_samples"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "multi_patient_summary.csv"
)


# --------------------------------------------------
# 2. Load summary
# --------------------------------------------------

df = pd.read_csv(SUMMARY_PATH)

results = []


# --------------------------------------------------
# 3. Inspect each window
# --------------------------------------------------

for _, row in df.iterrows():

    filename = row["filename"]
    patient_id = row["patient_id"]

    file_path = os.path.join(
        DATA_DIR,
        filename
    )

    data = np.load(file_path)

    mask = data["mask"]

    # Keep only Normal (0) and PVC (4) annotations
    valid_mask = mask[
        (mask == 0) | (mask == 4)
    ]

    if len(valid_mask) == 0:
        continue

    normal_count = np.sum(valid_mask == 0)
    pvc_count = np.sum(valid_mask == 4)

    total = normal_count + pvc_count

    pvc_ratio = pvc_count / total

    # Current prototype labelling rule
    if pvc_ratio > 0.5:
        rhythm = "PVC"
    else:
        rhythm = "Normal"

    results.append({
        "filename": filename,
        "patient_id": patient_id,
        "normal_count": normal_count,
        "pvc_count": pvc_count,
        "pvc_ratio": round(pvc_ratio, 4),
        "rhythm": rhythm
    })


# --------------------------------------------------
# 4. Create results table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nWINDOW-LEVEL LABEL DISTRIBUTION")
print("-" * 60)

print(
    results_df["rhythm"].value_counts()
)


# --------------------------------------------------
# 5. Patient-wise distribution
# --------------------------------------------------

print("\nPATIENT-WISE WINDOW LABELS")
print("-" * 60)

patient_counts = pd.crosstab(
    results_df["patient_id"],
    results_df["rhythm"]
)

print(patient_counts)


# --------------------------------------------------
# 6. Display PVC-containing windows
# --------------------------------------------------

print("\nPVC-CONTAINING WINDOWS")
print("-" * 60)

pvc_windows = results_df[
    results_df["pvc_count"] > 0
]

print(
    pvc_windows.to_string(index=False)
)


# --------------------------------------------------
# 7. Save detailed report
# --------------------------------------------------

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "window_label_distribution.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDetailed report saved!")
print(OUTPUT_PATH)