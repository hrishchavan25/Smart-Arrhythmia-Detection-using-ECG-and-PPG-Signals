
# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# LABELED DATASET INSPECTION
# ---------------------------------------------------------

from datasets import load_dataset
import numpy as np
import pandas as pd


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

print("Dataset stream initialized successfully!")


# ---------------------------------------------------------
# 2. INSPECT FIRST 100 WINDOWS
# ---------------------------------------------------------

MAX_SAMPLES = 100

records = []

for i, sample in enumerate(dataset):

    if i >= MAX_SAMPLES:
        break

    ecg = np.asarray(sample["ecg"], dtype=float)

    ppg = np.asarray(sample["ppg"], dtype=float)

    mask = np.asarray(sample["mask"], dtype=int)

    records.append({

        "window_id": sample["window_id"],

        "caseid": sample["caseid"],

        "split": sample["split"],

        "ecg": ecg,

        "ppg": ppg,

        "mask": mask

    })

    if (i + 1) % 20 == 0:

        print("Inspected", i + 1, "windows")


# ---------------------------------------------------------
# 3. DISPLAY DATASET INFORMATION
# ---------------------------------------------------------

print("\nDATASET INSPECTION COMPLETED!")

print("Number of inspected windows:", len(records))


# ---------------------------------------------------------
# 4. CHECK SIGNAL LENGTHS
# ---------------------------------------------------------

for i, record in enumerate(records[:5]):

    print("\nWindow:", record["window_id"])

    print("ECG length:", len(record["ecg"]))

    print("PPG length:", len(record["ppg"]))

    print("Mask length:", len(record["mask"]))

    print("Patient ID:", record["caseid"])


# ---------------------------------------------------------
# 5. CHECK AVAILABLE LABELS
# ---------------------------------------------------------

all_labels = []

for record in records:

    all_labels.extend(record["mask"].tolist())

unique_labels, counts = np.unique(
    all_labels,
    return_counts=True
)

print("\nAvailable label values:")

for label, count in zip(unique_labels, counts):

    print("Label:", label, "| Count:", count)


# ---------------------------------------------------------
# 6. CHECK ARRHYTHMIA CLASSES
# ---------------------------------------------------------

class_names = {
    0: "Normal",
    1: "Atrial Fibrillation",
    2: "Supraventricular Tachycardia",
    3: "Ventricular Tachycardia",
    4: "Premature Ventricular Contraction",
    5: "Premature Atrial Contraction",
    6: "Sinus Node Dysfunction",
    7: "Multifocal Atrial Tachycardia",
    8: "Atrioventricular Block"

}
print("\nLabel interpretation:")
for label in unique_labels:
    print(
        label,
        "->",
        class_names.get(label, "Unknown")
    )
# ---------------------------------------------------------
# 7. CREATE SUMMARY
# ---------------------------------------------------------
summary = pd.DataFrame({
    "window_id": [
        record["window_id"] for record in records
    ],
    "patient_id": [
        record["caseid"] for record in records
    ],
    "split": [
        record["split"] for record in records
    ],
    "ecg_length": [
        len(record["ecg"]) for record in records
    ],
    "ppg_length": [
        len(record["ppg"]) for record in records
    ],
    "mask_length": [
        len(record["mask"]) for record in records
    ]
})
# ---------------------------------------------------------
# 8. SAVE SUMMARY
# ---------------------------------------------------------
summary.to_csv(
    "dataset_inspection_summary.csv",
    index=False
)
print("\nSummary saved successfully!")
print("\nDATASET INSPECTION FINISHED!")