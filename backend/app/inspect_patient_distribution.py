
import os
import pandas as pd


# --------------------------------------------------
# 1. Set file paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "multi_patient_summary.csv"
)


# --------------------------------------------------
# 2. Load dataset summary
# --------------------------------------------------

df = pd.read_csv(SUMMARY_PATH)

print("\nDATASET SUMMARY")
print("-" * 50)

print("Total samples:", len(df))
print("Total patients:", df["patient_id"].nunique())


# --------------------------------------------------
# 3. Group samples by patient
# --------------------------------------------------

patient_summary = df.groupby("patient_id").agg(
    total_samples=("filename", "count"),
    normal_samples=("normal_count", lambda x: (x > 0).sum()),
    pvc_samples=("pvc_count", lambda x: (x > 0).sum()),
    total_normal=("normal_count", "sum"),
    total_pvc=("pvc_count", "sum")
).reset_index()


# --------------------------------------------------
# 4. Display patient-wise distribution
# --------------------------------------------------

print("\nPATIENT-WISE CLASS DISTRIBUTION")
print("-" * 70)

print(patient_summary.to_string(index=False))


# --------------------------------------------------
# 5. Identify patients containing both classes
# --------------------------------------------------

patient_summary["has_normal"] = (
    patient_summary["normal_samples"] > 0
)

patient_summary["has_pvc"] = (
    patient_summary["pvc_samples"] > 0
)

patient_summary["has_both_classes"] = (
    patient_summary["has_normal"]
    & patient_summary["has_pvc"]
)

print("\nPATIENTS WITH BOTH NORMAL AND PVC SAMPLES")
print("-" * 50)

print(
    patient_summary[
        patient_summary["has_both_classes"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 6. Save report
# --------------------------------------------------

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "patient_class_distribution.csv"
)

patient_summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nReport saved successfully!")
print(OUTPUT_PATH)