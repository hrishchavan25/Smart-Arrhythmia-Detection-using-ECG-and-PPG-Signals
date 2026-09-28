
# ---------------------------------------------------------
# SMART ARRHYTHMIA DETECTION SYSTEM
# ECG + PPG SIGNAL AND LABEL VISUALIZATION
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

NUM_SAMPLES_TO_PLOT = 3

NORMAL_LABEL = 0
PVC_LABEL = 4


# ---------------------------------------------------------
# 2. DEFINE DIRECTORIES
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

PLOT_DIR = os.path.join(
    DATA_DIR,
    "signal_visualizations"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# 3. GET SAMPLE FILES
# ---------------------------------------------------------

sample_files = sorted([

    file for file in os.listdir(SAMPLE_DIR)

    if file.endswith(".npz")

])

sample_files = sample_files[
    :NUM_SAMPLES_TO_PLOT
]

print("Selected samples:", sample_files)


# ---------------------------------------------------------
# 4. INITIALIZE SUMMARY
# ---------------------------------------------------------

summary_records = []


# ---------------------------------------------------------
# 5. PROCESS EACH SAMPLE
# ---------------------------------------------------------

for filename in sample_files:

    file_path = os.path.join(
        SAMPLE_DIR,
        filename
    )

    data = np.load(file_path)

    ecg = data["ecg"]

    ppg = data["ppg"]

    mask = data["mask"]

    # Verify signal alignment
    if not (
        len(ecg) == len(ppg) == len(mask)
    ):

        print(
            "Alignment error:",
            filename
        )

        continue

    # Create sample index
    sample_index = np.arange(
        len(ecg)
    )

    # Count annotations
    normal_count = np.sum(
        mask == NORMAL_LABEL
    )

    pvc_count = np.sum(
        mask == PVC_LABEL
    )

    other_count = len(mask) - (
        normal_count + pvc_count
    )

    normal_percentage = (
        normal_count / len(mask)
    ) * 100

    pvc_percentage = (
        pvc_count / len(mask)
    ) * 100

    unique_labels = np.unique(mask)

    # Extract patient and window details
    summary_row = {

        "filename": filename,

        "ecg_length": len(ecg),

        "ppg_length": len(ppg),

        "mask_length": len(mask),

        "normal_count": normal_count,

        "pvc_count": pvc_count,

        "other_count": other_count,

        "normal_percentage": normal_percentage,

        "pvc_percentage": pvc_percentage,

        "unique_labels": str(
            unique_labels.tolist()
        )

    }

    summary_records.append(
        summary_row
    )


    # -----------------------------------------------------
    # 6. CREATE FIGURE
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True
    )

    fig.suptitle(
        f"ECG + PPG Annotation Analysis: {filename}",
        fontsize=15
    )


    # -----------------------------------------------------
    # 7. ECG PLOT
    # -----------------------------------------------------

    axes[0].plot(
        sample_index,
        ecg,
        color="blue",
        linewidth=0.8
    )

    axes[0].set_title(
        "ECG Signal"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(
        True,
        alpha=0.3
    )


    # -----------------------------------------------------
    # 8. PPG PLOT
    # -----------------------------------------------------

    axes[1].plot(
        sample_index,
        ppg,
        color="red",
        linewidth=0.8
    )

    axes[1].set_title(
        "PPG Signal"
    )

    axes[1].set_ylabel(
        "Amplitude"
    )

    axes[1].grid(
        True,
        alpha=0.3
    )


    # -----------------------------------------------------
    # 9. ANNOTATION MASK
    # -----------------------------------------------------

    axes[2].step(
        sample_index,
        mask,
        where="post",
        color="black",
        linewidth=0.9
    )

    axes[2].set_title(
        "Sample-Level Rhythm Annotations"
    )

    axes[2].set_ylabel(
        "Label"
    )

    axes[2].set_xlabel(
        "Sample Number"
    )

    axes[2].set_yticks(
        sorted(unique_labels)
    )

    axes[2].grid(
        True,
        alpha=0.3
    )


    # -----------------------------------------------------
    # 10. HIGHLIGHT PVC REGIONS
    # -----------------------------------------------------

    axes[2].fill_between(

        sample_index,

        0,

        mask,

        where=(mask == PVC_LABEL),

        step="post",

        alpha=0.25,

        color="orange",

        label="PVC"

    )

    axes[2].legend()


    # -----------------------------------------------------
    # 11. FINALIZE FIGURE
    # -----------------------------------------------------

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_filename = (
        filename.replace(".npz", ".png")
    )

    output_path = os.path.join(
        PLOT_DIR,
        output_filename
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # -----------------------------------------------------
    # 12. DISPLAY SAMPLE SUMMARY
    # -----------------------------------------------------

    print("\n----------------------------------")

    print("SAMPLE:", filename)

    print("ECG length:", len(ecg))

    print("PPG length:", len(ppg))

    print("Mask length:", len(mask))

    print("Unique labels:", unique_labels)

    print("Normal samples:", normal_count)

    print("PVC samples:", pvc_count)

    print(
        "Normal percentage:",
        round(normal_percentage, 2)
    )

    print(
        "PVC percentage:",
        round(pvc_percentage, 2)
    )

    print("Plot saved:", output_path)


# ---------------------------------------------------------
# 13. SAVE SUMMARY
# ---------------------------------------------------------

summary_df = pd.DataFrame(
    summary_records
)

summary_path = os.path.join(
    DATA_DIR,
    "signal_visualization_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ---------------------------------------------------------
# 14. COMPLETION
# ---------------------------------------------------------

print("\n----------------------------------")

print("SIGNAL VISUALIZATION COMPLETED!")

print("----------------------------------")

print(
    "Total samples visualized:",
    len(summary_df)
)

print(
    "Plots saved in:",
    PLOT_DIR
)

print(
    "Summary saved to:",
    summary_path
)