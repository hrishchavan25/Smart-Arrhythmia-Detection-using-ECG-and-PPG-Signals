# --------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# ECG + PPG FEATURE EXTRACTION AND FUSION
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
# ---------------------------------------------------------
# 1. LOAD PREPROCESSED SIGNALS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_signals.npz"
)
data = np.load(DATA_PATH)
ecg_segments = data["ecg_segments"]
ppg_segments = data["ppg_segments"]
FS = int(data["sampling_frequency"])
print("Preprocessed signals loaded successfully!")
print("ECG segments:", ecg_segments.shape)
print("PPG segments:", ppg_segments.shape)
print("Sampling frequency:", FS)
# ---------------------------------------------------------
# 2. STATISTICAL FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_statistical_features(signal):
    features = {}
    features["mean"] = np.mean(signal)
    features["std"] = np.std(signal)
    features["variance"] = np.var(signal)
    features["minimum"] = np.min(signal)
    features["maximum"] = np.max(signal)
    features["range"] = np.ptp(signal)
    features["rms"] = np.sqrt(np.mean(signal ** 2))
    features["skewness"] = skew(signal)
    features["kurtosis"] = kurtosis(signal)
    return features
# ---------------------------------------------------------
# 3. ECG FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_ecg_features(signal, fs):
    features = {}
    # Statistical features
    statistical = extract_statistical_features(signal)
    for name, value in statistical.items():
        features["ecg_" + name] = value
    # Detect prominent positive R-peaks
    min_distance = int(0.30 * fs)
    peaks, properties = find_peaks(
        signal,
        distance=min_distance,
        prominence=0.5
    )
    features["ecg_peak_count"] = len(peaks)
    if len(peaks) >= 2:
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(peaks) / fs
        # Heart rate in beats per minute
        heart_rate = 60 / np.mean(rr_intervals)
        features["ecg_mean_rr"] = np.mean(rr_intervals)
        features["ecg_std_rr"] = np.std(rr_intervals)
        features["ecg_mean_hr"] = heart_rate
        features["ecg_sdnn"] = np.std(rr_intervals, ddof=1)
        if len(rr_intervals) >= 2:
            rr_differences = np.diff(rr_intervals)
            rmssd = np.sqrt(
                np.mean(rr_differences ** 2)
            )
            features["ecg_rmssd"] = rmssd
        else:
            features["ecg_rmssd"] = np.nan
    else:
        features["ecg_mean_rr"] = np.nan
        features["ecg_std_rr"] = np.nan
        features["ecg_mean_hr"] = np.nan
        features["ecg_sdnn"] = np.nan
        features["ecg_rmssd"] = np.nan
    return features
# ---------------------------------------------------------
# 4. PPG FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_ppg_features(signal, fs):
    features = {}
    # Statistical features
    statistical = extract_statistical_features(signal)
    for name, value in statistical.items():
        features["ppg_" + name] = value
    # Detect pulse peaks
    min_distance = int(0.30 * fs)
    peaks, properties = find_peaks(
        signal,
        distance=min_distance,
        prominence=0.3
    )
    features["ppg_peak_count"] = len(peaks)
    if len(peaks) >= 2:
        pulse_intervals = np.diff(peaks) / fs
        pulse_rate = 60 / np.mean(pulse_intervals)
        features["ppg_mean_interval"] = np.mean(pulse_intervals)
        features["ppg_std_interval"] = np.std(pulse_intervals)
        features["ppg_mean_rate"] = pulse_rate
        features["ppg_interval_variability"] = np.std(
            pulse_intervals
        )
    else:
        features["ppg_mean_interval"] = np.nan
        features["ppg_std_interval"] = np.nan
        features["ppg_mean_rate"] = np.nan
        features["ppg_interval_variability"] = np.nan
    return features
# ---------------------------------------------------------
# 5. MULTIMODAL FEATURE FUSION
# ---------------------------------------------------------
all_features = []
number_of_segments = min(
    len(ecg_segments),
    len(ppg_segments)
)
for i in range(number_of_segments):
    ecg = ecg_segments[i]
    ppg = ppg_segments[i]
    # Extract features separately
    ecg_features = extract_ecg_features(ecg, FS)
    ppg_features = extract_ppg_features(ppg, FS)
    # Combine both feature dictionaries
    combined_features = {}
    combined_features["segment_id"] = i + 1
    combined_features.update(ecg_features)
    combined_features.update(ppg_features)
    all_features.append(combined_features)
# ---------------------------------------------------------
# 6. CREATE FEATURE DATAFRAME
# ---------------------------------------------------------
feature_df = pd.DataFrame(all_features)
print("\nFEATURE EXTRACTION COMPLETED!")
print("\nFeature Matrix Shape:")
print(feature_df.shape)
print("\nExtracted Features:")
print(feature_df.columns.tolist())
print("\nFeature Matrix:")
print(feature_df)
# ---------------------------------------------------------
# 7. HANDLE INVALID VALUES
# ---------------------------------------------------------
feature_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)
print("\nMissing Feature Values:")
print(feature_df.isnull().sum())
# ---------------------------------------------------------
# 8. SAVE FEATURE MATRIX
# ---------------------------------------------------------
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ecg_ppg_features.csv"
)
feature_df.to_csv(
    OUTPUT_PATH,
    index=False
)
print("\nFeature matrix saved successfully!")
print("Output file:", OUTPUT_PATH)
# ---------------------------------------------------------
# END OF FEATURE EXTRACTION
# ---------------------------------------------------------
print("\nFEATURE EXTRACTION AND FUSION COMPLETED!")