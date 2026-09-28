# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# GAUSSIAN PROCESS UNCERTAINTY ANALYSIS
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
# 2. PREPARE INPUTS AND TARGETS
# ---------------------------------------------------------
X = df[["segment_id"]].to_numpy(dtype=float)
targets = {
    "ECG": df["ecg_mean_hr"].to_numpy(dtype=float),
    "PPG": df["ppg_mean_rate"].to_numpy(dtype=float)
}
# ---------------------------------------------------------
# 3. DEFINE KERNEL
# ---------------------------------------------------------
def create_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(
            length_scale=2.0,
            length_scale_bounds=(1e-2, 1e2)
        )
        + WhiteKernel(
            noise_level=0.1,
            noise_level_bounds=(1e-5, 1e1)
        )
    )
# ---------------------------------------------------------
# 4. LEAVE-ONE-OUT CROSS-VALIDATION
# ---------------------------------------------------------
def perform_loocv(X, y):
    actual_values = []
    predicted_values = []
    predictive_stds = []
    for i in range(len(X)):
        # Create training data without current segment
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        # Current segment becomes test data
        X_test = X[i].reshape(1, -1)
        y_test = y[i]
        # Create a fresh GP model
        model = GaussianProcessRegressor(
            kernel=create_kernel(),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42
        )
        # Train model
        model.fit(X_train, y_train)
        # Predict held-out segment
        prediction, std = model.predict(
            X_test,
            return_std=True
        )
        actual_values.append(y_test)
        predicted_values.append(prediction[0])
        predictive_stds.append(std[0])
    return (
        np.array(actual_values),
        np.array(predicted_values),
        np.array(predictive_stds)
    )
# ---------------------------------------------------------
# 5. CALCULATE EVALUATION METRICS
# ---------------------------------------------------------
def evaluate_predictions(actual, predicted, std):
    mae = mean_absolute_error(
        actual,
        predicted
    )
    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )
    # Approximate 95% predictive intervals
    lower = predicted - 1.96 * std
    upper = predicted + 1.96 * std
    # Check whether actual values fall inside intervals
    covered = (
        (actual >= lower)
        & (actual <= upper)
    )
    coverage = np.mean(covered) * 100
    # Average interval width
    interval_width = np.mean(upper - lower)
    # Standardized residuals
    standardized_residuals = (
        actual - predicted
    ) / np.maximum(std, 1e-10)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Coverage": coverage,
        "Mean_Interval_Width": interval_width,
        "Standardized_Residuals": standardized_residuals,
        "Lower_Bound": lower,
        "Upper_Bound": upper
    }
# ---------------------------------------------------------
# 6. RUN ANALYSIS FOR ECG AND PPG
# ---------------------------------------------------------
all_results = {}
for signal_name, y in targets.items():
    print("\n----------------------------------")
    print(signal_name, "UNCERTAINTY ANALYSIS")
    print("----------------------------------")
    actual, predicted, std = perform_loocv(X, y)
    metrics = evaluate_predictions(
        actual,
        predicted,
        std
    )
    all_results[signal_name] = {
        "actual": actual,
        "predicted": predicted,
        "std": std,
        "metrics": metrics
    }
    print("\nMean Absolute Error:", round(metrics["MAE"], 4))
    print("Root Mean Squared Error:", round(metrics["RMSE"], 4))
    print("95% Interval Coverage:", round(metrics["Coverage"], 2), "%")
    print(
        "Mean Interval Width:",
        round(metrics["Mean_Interval_Width"], 4)
    )
# ---------------------------------------------------------
# 7. CREATE RESULTS DATAFRAME
# ---------------------------------------------------------
results_df = pd.DataFrame({
    "segment_id": df["segment_id"],
    "ecg_actual": all_results["ECG"]["actual"],
    "ecg_predicted": all_results["ECG"]["predicted"],
    "ecg_predictive_std": all_results["ECG"]["std"],
    "ecg_lower_bound": all_results["ECG"]["metrics"]["Lower_Bound"],
    "ecg_upper_bound": all_results["ECG"]["metrics"]["Upper_Bound"],
    "ppg_actual": all_results["PPG"]["actual"],
    "ppg_predicted": all_results["PPG"]["predicted"],
    "ppg_predictive_std": all_results["PPG"]["std"],
    "ppg_lower_bound": all_results["PPG"]["metrics"]["Lower_Bound"],
    "ppg_upper_bound": all_results["PPG"]["metrics"]["Upper_Bound"]
})
# ---------------------------------------------------------
# 8. VISUALIZE ECG VALIDATION
# ---------------------------------------------------------
ecg_result = all_results["ECG"]
plt.figure(figsize=(12, 6))
plt.errorbar(
    X.ravel(),
    ecg_result["predicted"],
    yerr=1.96 * ecg_result["std"],
    fmt="o",
    capsize=5,
    color="red",
    label="LOOCV Prediction with 95% Interval"
)
plt.plot(
    X.ravel(),
    ecg_result["actual"],
    "bo-",
    label="Actual ECG Heart Rate"
)
plt.title("ECG Gaussian Process - LOOCV Validation")
plt.xlabel("Segment Number")
plt.ylabel("Heart Rate (BPM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 9. VISUALIZE PPG VALIDATION
# ---------------------------------------------------------
ppg_result = all_results["PPG"]
plt.figure(figsize=(12, 6))
plt.errorbar(
    X.ravel(),
    ppg_result["predicted"],
    yerr=1.96 * ppg_result["std"],
    fmt="o",
    capsize=5,
    color="orange",
    label="LOOCV Prediction with 95% Interval"
)
plt.plot(
    X.ravel(),
    ppg_result["actual"],
    "go-",
    label="Actual PPG Pulse Rate"
)
plt.title("PPG Gaussian Process - LOOCV Validation")
plt.xlabel("Segment Number")
plt.ylabel("Pulse Rate (BPM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 10. SAVE RESULTS
# ---------------------------------------------------------
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "uncertainty_analysis.csv"
)
results_df.to_csv(
    OUTPUT_PATH,
    index=False
)
print("\nUncertainty analysis saved successfully!")
print("Output file:", OUTPUT_PATH)
print("\nUNCERTAINTY ANALYSIS COMPLETED!")