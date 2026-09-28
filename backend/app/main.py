
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

from .signal_loader import load_ecg_ppg_file

app = FastAPI(
    title="CardioSense API",
    description="ECG and PPG Arrhythmia Detection System",
    version="1.0"
)

# Allow React frontend to communicate with Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "CardioSense API is running",
        "status": "online"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CardioSense Backend"
    }


# --------------------------------------------------
# Dashboard data
# --------------------------------------------------

@app.get("/api/dashboard")
def get_dashboard():

    # Temporary sample waveforms.
    # These will be replaced with actual dataset signals.

    ecg = [
        float(np.sin(i * 0.35) * 0.08 +
        (1.5 if i % 25 == 12 else 0))
        for i in range(100)
    ]

    ppg = [
        float(max(0, np.sin(i * 0.22)) * 0.9)
        for i in range(100)
    ]

    return {
        "signals": {
            "ecg": ecg,
            "ppg": ppg
        },
        "statistics": {
            "total_signals": 1284,
            "normal": 1106,
            "pvc": 178,
            "confidence": 94.6
        },
        "detections": [
            {
                "id": "CS-1024",
                "patient": "Patient 001",
                "rhythm": "Normal",
                "confidence": 97.8,
                "uncertainty": "Low",
                "time": "10:42 AM"
            },
            {
                "id": "CS-1023",
                "patient": "Patient 002",
                "rhythm": "PVC",
                "confidence": 91.4,
                "uncertainty": "Moderate",
                "time": "10:36 AM"
            }
        ]
    }


# --------------------------------------------------
# Signal analysis request
# --------------------------------------------------

class SignalRequest(BaseModel):
    ecg: list[float]
    ppg: list[float]


@app.post("/api/analyze")
def analyze_signals(data: SignalRequest):

    if len(data.ecg) == 0 or len(data.ppg) == 0:
        return {
            "error": "ECG and PPG signals are required"
        }

    return {
        "message": "Signals received successfully",
        "ecg_samples": len(data.ecg),
        "ppg_samples": len(data.ppg),
        "status": "received"
    }


@app.post("/api/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty."
        )

    try:
        signals = load_ecg_ppg_file(content)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    total_samples = signals["total_samples"]
    sample_indices = np.linspace(
        0,
        total_samples - 1,
        num=min(total_samples, 1000),
        dtype=int
    )

    return {
        "status": "received",
        "message": "Signals received successfully",
        "total_samples": total_samples,
        "ecg_column": signals["ecg_column"],
        "ppg_column": signals["ppg_column"],
        "ecg": np.asarray(signals["ecg"])[sample_indices].tolist(),
        "ppg": np.asarray(signals["ppg"])[sample_indices].tolist()
    }