
# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# GAUSSIAN PROCESS ARRHYTHMIA DETECTION
# ---------------------------------------------------------

import os
import glob
import joblib
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

PVC_LABEL = 4
NORMAL_LABEL = 0

PVC_THRESHOLD = 0.50

RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

SAMPLE_DIR = os.path.join(
    DATA_DIR,
    "multi_patient_samples"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# 3. FEATURE EXTRACTION
# ---------------------------------------------------------

def extract_features(signal, prefix):

    signal = np.asarray(
        signal,
        dtype=float
    )

    features = {}

    features[f"{prefix}_mean"] = np.mean(signal)

    features[f"{prefix}_std"] = np.std(signal)

    features[f"{prefix}_variance"] = np.var(signal)

    features[f"{prefix}_minimum"] = np.min(signal)

    features[f"{prefix}_maximum"] = np.max(signal)

    features[f"{prefix}_range"] = (
        np.max(signal) - np.min(signal)
    )

    features[f"{prefix}_rms"] = np.sqrt(
        np.mean(signal ** 2)
    )

    features[f"{prefix}_skewness"] = skew(
        signal
    )

    features[f"{prefix}_kurtosis"] = kurtosis(
        signal
    )

    return features


# ---------------------------------------------------------
# 4. LOAD DATA AND BUILD FEATURE DATAFRAME
# ---------------------------------------------------------

print("Loading multi-patient samples...")

sample_files = sorted(
    glob.glob(
        os.path.join(
            SAMPLE_DIR,
            "*.npz"
        )
    )
)

records = []

for file_path in sample_files:

    data = np.load(file_path)

    ecg = data["ecg"]

    ppg = data["ppg"]

    mask = data["mask"]

    filename = os.path.basename(
        file_path
    )

    # ---------------------------------------------
    # Calculate label proportions
    # ---------------------------------------------

    pvc_count = np.sum(
        mask == PVC_LABEL
    )

    normal_count = np.sum(
        mask == NORMAL_LABEL
    )

    total_labeled = (
        pvc_count + normal_count
    )

    if total_labeled == 0:
        continue

    pvc_ratio = (
        pvc_count / total_labeled
    )

    # ---------------------------------------------
    # Assign provisional window label
    # ---------------------------------------------

    if pvc_ratio > PVC_THRESHOLD:

        label = 1

        rhythm = "PVC"

    else:

        label = 0

        rhythm = "Normal"

    # ---------------------------------------------
    # Extract ECG and PPG features
    # ---------------------------------------------

    features = {}

    features.update(
        extract_features(
            ecg,
            "ecg"
        )
    )

    features.update(
        extract_features(
            ppg,
            "ppg"
        )
    )

    # ---------------------------------------------
    # Save record
    # ---------------------------------------------

    features["filename"] = filename

    features["pvc_ratio"] = pvc_ratio

    features["label"] = label

    features["rhythm"] = rhythm

    records.append(features)


df = pd.DataFrame(records)


# ---------------------------------------------------------
# 5. MERGE PATIENT INFORMATION
# ---------------------------------------------------------

summary_path = os.path.join(
    DATA_DIR,
    "multi_patient_summary.csv"
)

summary_df = pd.read_csv(
    summary_path
)

df = df.merge(
    summary_df[
        ["filename", "patient_id", "split"]
    ],
    on="filename",
    how="left"
)

print("\nFeature extraction completed!")

print("Dataset shape:", df.shape)

print("\nClass distribution:")

print(
    df["rhythm"].value_counts()
)


# ---------------------------------------------------------
# 6. PREPARE MODEL INPUT
# ---------------------------------------------------------

excluded_columns = [

    "filename",

    "patient_id",

    "split",

    "pvc_ratio",

    "label",

    "rhythm"

]

X = df.drop(
    columns=excluded_columns
)

y = df["label"]

groups = df["patient_id"]


# ---------------------------------------------------------
# 7. PATIENT-INDEPENDENT TRAIN-TEST SPLIT
# ---------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_index, test_index = next(
    splitter.split(
        X,
        y,
        groups
    )
)

X_train = X.iloc[train_index]

X_test = X.iloc[test_index]

y_train = y.iloc[train_index]

y_test = y.iloc[test_index]

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))

print(
    "Training patients:",
    groups.iloc[train_index].nunique()
)

print(
    "Testing patients:",
    groups.iloc[test_index].nunique()
)


# ---------------------------------------------------------
# 8. BUILD GAUSSIAN PROCESS CLASSIFIER
# ---------------------------------------------------------

kernel = (
    ConstantKernel(1.0)
    * RBF(length_scale=1.0)
)

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        GaussianProcessClassifier(
            kernel=kernel,
            random_state=RANDOM_STATE,
            max_iter_predict=100
        )
    )

])


# ---------------------------------------------------------
# 9. TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining Gaussian Process Classifier...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ---------------------------------------------------------
# 10. PREDICTIONS
# ---------------------------------------------------------

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)


# ---------------------------------------------------------
# 11. UNCERTAINTY ESTIMATION
# ---------------------------------------------------------

# Binary classification entropy
epsilon = 1e-10

entropy = -np.sum(

    probabilities
    * np.log(
        probabilities + epsilon
    ),

    axis=1

)

# Normalize entropy to range 0 to 1
normalized_uncertainty = (
    entropy / np.log(2)
)


# ---------------------------------------------------------
# 12. EVALUATION
# ---------------------------------------------------------

print("\nMODEL EVALUATION")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "Normal",
            "PVC"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)


# ---------------------------------------------------------
# 13. SAVE PREDICTIONS
# ---------------------------------------------------------

results = df.iloc[
    test_index
][
    [
        "filename",
        "patient_id",
        "rhythm",
        "pvc_ratio"
    ]
].copy()

results["actual_label"] = y_test.values

results["predicted_label"] = predictions

results["normal_probability"] = (
    probabilities[:, 0]
)

results["pvc_probability"] = (
    probabilities[:, 1]
)

results["uncertainty"] = (
    normalized_uncertainty
)

results["predicted_rhythm"] = np.where(
    predictions == 1,
    "PVC",
    "Normal"
)

results_path = os.path.join(
    DATA_DIR,
    "detection_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ---------------------------------------------------------
# 14. SAVE TRAINED MODEL
# ---------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "gaussian_process_detector.pkl"
)

joblib.dump(
    {
        "model": model,
        "feature_columns": list(X.columns)
    },
    model_path
)


# ---------------------------------------------------------
# 15. COMPLETION
# ---------------------------------------------------------

print("\n----------------------------------")

print("ARRHYTHMIA DETECTION COMPLETED!")

print("----------------------------------")

print("Model saved:", model_path)

print("Results saved:", results_path)

print("\nDetection module finished!")