
# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# MULTI-PATIENT DATASET SAMPLING
# ---------------------------------------------------------

from datasets import load_dataset
import numpy as np
import pandas as pd
import os


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

MAX_ENTRIES = 10000

TARGET_PATIENTS = 10

WINDOWS_PER_PATIENT = 10

TOTAL_TARGET_WINDOWS = (
    TARGET_PATIENTS * WINDOWS_PER_PATIENT
)


# ---------------------------------------------------------
# 2. DEFINE OUTPUT DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "multi_patient_samples"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# 3. LOAD DATASET
# ---------------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "sukju/cardiologent_dataset",
    "waveforms",
    split="train",
    streaming=True
)

print("Dataset loaded successfully!")


# ---------------------------------------------------------
# 4. INITIALIZE PATIENT STORAGE
# ---------------------------------------------------------

patient_records = {}

entries_scanned = 0


# ---------------------------------------------------------
# 5. COLLECT WINDOWS FROM DIFFERENT PATIENTS
# ---------------------------------------------------------

print("\nCollecting multi-patient samples...")

for sample in dataset:

    if entries_scanned >= MAX_ENTRIES:
        break

    entries_scanned += 1

    patient_id = str(sample["caseid"])

    # Create storage for a new patient
    if patient_id not in patient_records:

        # Stop if target patient count is reached
        if len(patient_records) >= TARGET_PATIENTS:
            continue

        patient_records[patient_id] = []

    # Skip patients who already have enough windows
    if len(patient_records[patient_id]) >= WINDOWS_PER_PATIENT:
        continue

    # Preserve ECG, PPG and annotations
    ecg = np.asarray(
        sample["ecg"],
        dtype=np.float32
    )

    ppg = np.asarray(
        sample["ppg"],
        dtype=np.float32
    )

    mask = np.asarray(
        sample["mask"],
        dtype=np.int32
    )

    # Verify alignment
    if not (
        len(ecg) == len(ppg) == len(mask)
    ):
        print(
            "Skipping misaligned window:",
            sample["window_id"]
        )
        continue

    record = {

        "window_id": str(sample["window_id"]),

        "patient_id": patient_id,

        "split": str(sample["split"]),

        "ecg": ecg,

        "ppg": ppg,

        "mask": mask

    }

    patient_records[patient_id].append(record)

    # Display progress
    if (
        len(patient_records[patient_id])
        == WINDOWS_PER_PATIENT
    ):

        print(
            "Patient",
            patient_id,
            "completed:",
            WINDOWS_PER_PATIENT,
            "windows"
        )

    # Stop once all selected patients have enough windows
    if (
        len(patient_records) == TARGET_PATIENTS
        and all(
            len(windows) == WINDOWS_PER_PATIENT
            for windows in patient_records.values()
        )
    ):
        break


# ---------------------------------------------------------
# 6. SAVE WAVEFORMS
# ---------------------------------------------------------

print("\nSaving waveform data...")

summary_records = []

sample_number = 0

for patient_id, windows in patient_records.items():

    for record in windows:

        sample_number += 1

        filename = (
            f"sample_{sample_number:03d}.npz"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        # Save variable-length arrays individually
        np.savez_compressed(

            output_path,

            ecg=record["ecg"],

            ppg=record["ppg"],

            mask=record["mask"]

        )

        # Calculate annotation counts
        unique_labels, counts = np.unique(
            record["mask"],
            return_counts=True
        )

        label_counts = dict(
            zip(
                unique_labels.astype(int),
                counts.astype(int)
            )
        )

        summary_records.append({

            "sample_number": sample_number,

            "filename": filename,

            "window_id": record["window_id"],

            "patient_id": patient_id,

            "split": record["split"],

            "ecg_length": len(record["ecg"]),

            "ppg_length": len(record["ppg"]),

            "mask_length": len(record["mask"]),

            "normal_count": label_counts.get(0, 0),

            "pvc_count": label_counts.get(4, 0),

            "unique_labels": len(unique_labels)

        })


# ---------------------------------------------------------
# 7. CREATE SUMMARY DATAFRAME
# ---------------------------------------------------------

summary_df = pd.DataFrame(
    summary_records
)


# ---------------------------------------------------------
# 8. SAVE SUMMARY CSV
# ---------------------------------------------------------

SUMMARY_PATH = os.path.join(
    DATA_DIR,
    "multi_patient_summary.csv"
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)


# ---------------------------------------------------------
# 9. DISPLAY FINAL RESULTS
# ---------------------------------------------------------

print("\n----------------------------------")

print("MULTI-PATIENT SAMPLING COMPLETED!")

print("----------------------------------")

print(
    "Entries scanned:",
    entries_scanned
)

print(
    "Unique patients selected:",
    summary_df["patient_id"].nunique()
)

print(
    "Total windows collected:",
    len(summary_df)
)

print("\nPATIENT-WISE WINDOW COUNTS")

print(
    summary_df.groupby(
        "patient_id"
    )["window_id"].count()
)

print("\nCLASS DISTRIBUTION")

print(
    "Normal samples:",
    summary_df["normal_count"].sum()
)

print(
    "PVC samples:",
    summary_df["pvc_count"].sum()
)

print("\nSummary saved to:")

print(SUMMARY_PATH)

print("\nWaveforms saved in:")

print(OUTPUT_DIR)

print("\nSAMPLING FINISHED!")