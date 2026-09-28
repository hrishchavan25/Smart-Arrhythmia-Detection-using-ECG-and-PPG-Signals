
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CardioSense API")

# Allow the React frontend to communicate with the backend
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


@app.get("/")
def home():
    return {
        "message": "CardioSense backend is running!"
    }


@app.get("/api/model-results")
def model_results():
    return {
        "status": "success",
        "gaussian": {
            "mean": [150, 140, 125, 95, 65, 80, 115, 145, 130, 100, 85, 110],
            "upper": [170, 165, 155, 130, 105, 125, 155, 180, 170, 145, 135, 160],
            "lower": [130, 115, 95, 60, 25, 35, 75, 110, 90, 55, 35, 60]
        },
        "bayesian": {
            "before_calibration": [0.04, 0.12, 0.19, 0.27, 0.35, 0.46, 0.55, 0.65, 0.76],
            "after_calibration": [0.09, 0.19, 0.31, 0.39, 0.51, 0.59, 0.69, 0.79, 0.89]
        }
    }