from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_single
from fastapi.middleware.cors import CORSMiddleware


# ----------------------------------------
# FastAPI application
# ----------------------------------------

app = FastAPI(
    title="Flight Delay Prediction API",
    description="LightGBM-based flight delay prediction service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Request schema
# ----------------------------------------

class FlightRequest(BaseModel):
    FlightDate: str
    Reporting_Airline: str
    Origin: str
    Dest: str
    CRSDepTime: int
    CRSArrTime: int
    CRSElapsedTime: float
    Distance: float


# ----------------------------------------
# Health check
# ----------------------------------------

@app.get("/")
def root():
    return {
        "message": "Flight Delay Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ----------------------------------------
# Prediction endpoint
# ----------------------------------------

@app.post("/predict")
def predict(request: FlightRequest):

    flight_data = request.model_dump()

    result = predict_single(flight_data)

    return result