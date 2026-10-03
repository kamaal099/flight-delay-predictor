import joblib
import pandas as pd

from src.feature_engineering import create_features
from src.prepare_model_data import prepare_for_lightgbm


MODEL_PATH = "models/flight_delay_lgbm.pkl"

# Optimized threshold from evaluation
THRESHOLD = 0.38


def load_model():
    """Load trained LightGBM model."""
    return joblib.load(MODEL_PATH)


def predict_delay(df: pd.DataFrame):
    """
    Predict whether flights will be delayed.

    Returns:
        predictions: 0 = On Time, 1 = Delayed
        probabilities: probability of delay
    """

    model = load_model()

    # Feature engineering
    df_features = create_features(df.copy())

    # Prepare features
    X = df_features.copy()

    # We don't have y here, so prepare manually if necessary.
    # Keep only the columns expected by the trained model.
    X, _ = prepare_for_lightgbm(X, X.copy())

    # Probability of delay
    probabilities = model.predict_proba(X)[:, 1]

    # Optimized threshold
    predictions = (probabilities >= THRESHOLD).astype(int)

    return predictions, probabilities


def predict_single(flight_data: dict):
    """
    Predict delay for a single flight.
    """

    df = pd.DataFrame([flight_data])

    predictions, probabilities = predict_delay(df)

    prediction = int(predictions[0])
    probability = float(probabilities[0])

    return {
        "prediction": prediction,
        "status": "Delayed" if prediction == 1 else "On Time",
        "delay_probability": round(probability, 4),
        "threshold": THRESHOLD
    }