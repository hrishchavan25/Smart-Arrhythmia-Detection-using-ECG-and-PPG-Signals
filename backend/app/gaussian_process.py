# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# GAUSSIAN PROCESS REGRESSION
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel
)
# ---------------------------------------------------------
# 1. LOAD FEATURE MATRIX
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ecg_ppg_features.csv"
)
df = pd.read_csv(DATA_PATH)
print("Feature matrix loaded successfully!")
print("Dataset shape:", df.shape)
# ---------------------------------------------------------
# 2. PREPARE INPUT AND TARGET
# ---------------------------------------------------------
# Segment number is our input
X = df[["segment_id"]].to_numpy(dtype=float)
# ECG mean heart rate is our first target
y_ecg = df["ecg_mean_hr"].to_numpy(dtype=float)
# PPG mean pulse rate is our second target
y_ppg = df["ppg_mean_rate"].to_numpy(dtype=float)
print("\nInput shape:", X.shape)
print("ECG target shape:", y_ecg.shape)
print("PPG target shape:", y_ppg.shape)
# ---------------------------------------------------------
# 3. DEFINE GAUSSIAN PROCESS KERNEL
# ---------------------------------------------------------
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=2.0, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
)
print("\nKernel configuration:")
print(kernel)
# ---------------------------------------------------------
# 4. CREATE GAUSSIAN PROCESS MODEL
# ---------------------------------------------------------
gp_ecg = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=5,
    random_state=42
)
gp_ppg = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=5,
    random_state=42
)
# ---------------------------------------------------------
# 5. TRAIN GAUSSIAN PROCESS MODELS
# ---------------------------------------------------------
gp_ecg.fit(X, y_ecg)
gp_ppg.fit(X, y_ppg)
print("\nGaussian Process models trained successfully!")
# ---------------------------------------------------------
# 6. GENERATE PREDICTIONS
# ---------------------------------------------------------
# Generate smooth input points for visualization
X_test = np.linspace(
    X.min(),
    X.max(),
    200
).reshape(-1, 1)
# ECG predictions
ecg_mean, ecg_std = gp_ecg.predict(
    X_test,
    return_std=True
)
# PPG predictions
ppg_mean, ppg_std = gp_ppg.predict(
    X_test,
    return_std=True
)
# ---------------------------------------------------------
# 7. CALCULATE PREDICTIVE VARIANCE
# ---------------------------------------------------------
ecg_variance = ecg_std ** 2
ppg_variance = ppg_std ** 2
print("\nPredictive uncertainty calculated!")
# ---------------------------------------------------------
# 8. DISPLAY SAMPLE PREDICTIONS
# ---------------------------------------------------------
print("\nECG Predictions:")
for i in range(0, 200, 25):
    print(
        "Segment:",
        round(X_test[i][0], 2),
        "| Mean:",
        round(ecg_mean[i], 2),
        "| Standard Deviation:",
        round(ecg_std[i], 4),
        "| Variance:",
        round(ecg_variance[i], 6)
    )
print("\nPPG Predictions:")
for i in range(0, 200, 25):
    print(
        "Segment:",
        round(X_test[i][0], 2),
        "| Mean:",
        round(ppg_mean[i], 2),
        "| Standard Deviation:",
        round(ppg_std[i], 4),
        "| Variance:",
        round(ppg_variance[i], 6)
    )
# ---------------------------------------------------------
# 9. VISUALIZE ECG PREDICTIONS
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(
    X,
    y_ecg,
    color="blue",
    label="Observed ECG Heart Rate"
)
plt.plot(
    X_test,
    ecg_mean,
    color="red",
    label="GP Predictive Mean"
)
plt.fill_between(
    X_test.ravel(),
    ecg_mean - 1.96 * ecg_std,
    ecg_mean + 1.96 * ecg_std,
    color="red",
    alpha=0.2,
    label="Approximate 95% Predictive Interval"
)
plt.title("Gaussian Process Regression - ECG")
plt.xlabel("Segment Number")
plt.ylabel("Heart Rate (BPM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 10. VISUALIZE PPG PREDICTIONS
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(
    X,
    y_ppg,
    color="green",
    label="Observed PPG Pulse Rate"
)
plt.plot(
    X_test,
    ppg_mean,
    color="orange",
    label="GP Predictive Mean"
)
plt.fill_between(
    X_test.ravel(),
    ppg_mean - 1.96 * ppg_std,
    ppg_mean + 1.96 * ppg_std,
    color="orange",
    alpha=0.2,
    label="Approximate 95% Predictive Interval"
)
plt.title("Gaussian Process Regression - PPG")
plt.xlabel("Segment Number")
plt.ylabel("Pulse Rate (BPM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 11. SAVE PREDICTIONS
# ---------------------------------------------------------
prediction_df = pd.DataFrame({
    "segment_id": X_test.ravel(),
    "ecg_predictive_mean": ecg_mean,
    "ecg_predictive_std": ecg_std,
    "ecg_predictive_variance": ecg_variance,
    "ppg_predictive_mean": ppg_mean,
    "ppg_predictive_std": ppg_std,
    "ppg_predictive_variance": ppg_variance
})
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "gp_predictions.csv"
)
prediction_df.to_csv(
    OUTPUT_PATH,
    index=False
)
print("\nPredictions saved successfully!")
print("Output file:", OUTPUT_PATH)
# ---------------------------------------------------------
# END OF GAUSSIAN PROCESS IMPLEMENTATION
# ---------------------------------------------------------
print("\nGAUSSIAN PROCESS MODELLING COMPLETED!")