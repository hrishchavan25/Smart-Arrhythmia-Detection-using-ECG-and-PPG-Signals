
# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# PATIENT DISTRIBUTION ANALYSIS
# ---------------------------------------------------------

from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

print("Loading CardioBench dataset...")

dataset = load_dataset(
    "sukju/cardiologent_dataset",
    "waveforms",
    split="train",
    streaming=True
)

print("Dataset loaded successfully!")


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

    normal_count = np.sum(mask == 0)

    pvc_count = np.sum(mask == 4)

    records.append({

        "window_id": sample["window_id"],

        "patient_id": str(sample["caseid"]),

        "split": sample["split"],

        "total_samples": len(mask),

        "normal_count": normal_count,

        "pvc_count": pvc_count,

        "normal_percentage": (
            normal_count / len(mask)
        ) * 100,

        "pvc_percentage": (
            pvc_count / len(mask)
        ) * 100,

        "contains_normal": normal_count > 0,

        "contains_pvc": pvc_count > 0

    })


# ---------------------------------------------------------
# 3. CREATE WINDOW DATAFRAME
# ---------------------------------------------------------

window_df = pd.DataFrame(records)

print("\nWINDOW DATA LOADED")

print("Total windows:", len(window_df))


# ---------------------------------------------------------
# 4. PATIENT-LEVEL SUMMARY
# ---------------------------------------------------------

patient_df = window_df.groupby(
    "patient_id"
).agg(

    total_windows=("window_id", "count"),

    total_samples=("total_samples", "sum"),

    normal_samples=("normal_count", "sum"),

    pvc_samples=("pvc_count", "sum")

).reset_index()


# ---------------------------------------------------------
# 5. CALCULATE PATIENT-LEVEL PERCENTAGES
# ---------------------------------------------------------

patient_df["normal_percentage"] = (

    patient_df["normal_samples"]

    / patient_df["total_samples"]

) * 100


patient_df["pvc_percentage"] = (

    patient_df["pvc_samples"]

    / patient_df["total_samples"]

) * 100


# ---------------------------------------------------------
# 6. IDENTIFY PATIENT CLASS COMPOSITION
# ---------------------------------------------------------

patient_df["has_normal"] = (
    patient_df["normal_samples"] > 0
)

patient_df["has_pvc"] = (
    patient_df["pvc_samples"] > 0
)


patient_df["patient_category"] = np.select(

    [
        patient_df["has_normal"]
        & patient_df["has_pvc"],

        patient_df["has_normal"]
        & ~patient_df["has_pvc"],

        ~patient_df["has_normal"]
        & patient_df["has_pvc"]

    ],

    [
        "Mixed",
        "Normal Only",
        "PVC Only"
    ],

    default="Other"

)


# ---------------------------------------------------------
# 7. DISPLAY RESULTS
# ---------------------------------------------------------

print("\nPATIENT DISTRIBUTION")

print(
    "Unique patients:",
    patient_df["patient_id"].nunique()
)

print(
    "Total windows:",
    patient_df["total_windows"].sum()
)


print("\nPATIENT SUMMARY")

print(
    patient_df.to_string(index=False)
)


# ---------------------------------------------------------
# 8. DISPLAY CATEGORY COUNTS
# ---------------------------------------------------------

print("\nPATIENT CATEGORY COUNTS")

print(
    patient_df["patient_category"].value_counts()
)


# ---------------------------------------------------------
# 9. VISUALIZE WINDOWS PER PATIENT
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(

    patient_df["patient_id"],

    patient_df["total_windows"]

)

plt.title("Number of Windows per Patient")

plt.xlabel("Patient ID")

plt.ylabel("Number of Windows")

plt.xticks(rotation=45)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 10. VISUALIZE CLASS DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(

    patient_df["patient_id"],

    patient_df["normal_percentage"],

    label="Normal"

)

plt.bar(

    patient_df["patient_id"],

    patient_df["pvc_percentage"],

    bottom=patient_df["normal_percentage"],

    label="PVC"

)

plt.title("Normal and PVC Distribution per Patient")

plt.xlabel("Patient ID")

plt.ylabel("Percentage of Annotated Samples")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 11. SAVE RESULTS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


window_path = os.path.join(
    DATA_DIR,
    "patient_window_summary.csv"
)
patient_path = os.path.join(
    DATA_DIR,
    "patient_distribution.csv"
)
window_df.to_csv(
    window_path,
    index=False
)
patient_df.to_csv(
    patient_path,
    index=False
)
# ---------------------------------------------------------
# 12. COMPLETION
# ---------------------------------------------------------
print("\nPatient analysis saved successfully!")
print("Window summary:", window_path)
print("Patient summary:", patient_path)
print("\nPATIENT DISTRIBUTION ANALYSIS COMPLETED!")