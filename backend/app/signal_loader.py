import pandas as pd
import numpy as np
from io import BytesIO
def find_column(columns, possible_names):

    for column in columns:

        column_lower = column.lower().strip()

        for name in possible_names:

            if name in column_lower:
                return column

    return None
def load_ecg_ppg_file(file_content):

    # Read CSV
    dataframe = pd.read_csv(
        BytesIO(file_content)
    )
    # Check empty file
    if dataframe.empty:
        raise ValueError("The CSV file is empty.")
    columns = dataframe.columns.tolist()

    # Possible ECG column names
    ecg_names = [
        "ecg",
        "electrocardiogram"
    ]

    # Possible PPG column names
    ppg_names = [
        "ppg",
        "photoplethysmogram",
        "photoplethysmography"
    ]

    # Possible time column names
    time_names = [
        "time",
        "timestamp",
        "sample_time"
    ]

    ecg_column = find_column(
        columns,
        ecg_names
    )

    ppg_column = find_column(
        columns,
        ppg_names
    )

    time_column = find_column(
        columns,
        time_names
    )

    # Validate ECG
    if ecg_column is None:

        raise ValueError(
            "ECG column could not be identified. "
            "Please check the CSV column names."
        )

    # Validate PPG
    if ppg_column is None:

        raise ValueError(
            "PPG column could not be identified. "
            "Please check the CSV column names."
        )

    # Extract ECG
    ecg = pd.to_numeric(
        dataframe[ecg_column],
        errors="coerce"
    )

    # Extract PPG
    ppg = pd.to_numeric(
        dataframe[ppg_column],
        errors="coerce"
    )

    # Remove invalid rows
    valid_rows = (
        ecg.notna() &
        ppg.notna()
    )

    ecg = ecg[valid_rows]
    ppg = ppg[valid_rows]

    if len(ecg) == 0:

        raise ValueError(
            "No valid ECG/PPG samples were found."
        )

    # Create response
    result = {

        "ecg": ecg.tolist(),

        "ppg": ppg.tolist(),

        "columns": columns,

        "ecg_column": str(ecg_column),

        "ppg_column": str(ppg_column),

        "time_column": (
            str(time_column)
            if time_column
            else None
        ),

        "total_samples": len(ecg)
    }

    return result