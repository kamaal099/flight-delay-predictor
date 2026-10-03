# ✈️ Flight Delay Predictor

An end-to-end machine learning system that predicts the probability of a commercial flight being delayed using **LightGBM**, **FastAPI**, and an interactive web dashboard.

The project combines data preprocessing, feature engineering, machine learning, threshold optimization, model evaluation, REST API development, and a modern frontend into a complete deployable prediction system.
---

## 🎯 Project Overview

Flight delays are affected by multiple factors including airline schedules, departure and arrival times, routes, travel distance, day of the week, and seasonal patterns.

This project uses historical flight data to build a machine learning model capable of estimating whether a flight is likely to be delayed.

The system takes flight information as input and returns:

- ✈️ Predicted flight status
- 📊 Probability of delay
- 🎚️ Optimized classification threshold
- 🛫 Route information
- 🏢 Airline information

The final system is exposed through a **FastAPI REST API** and connected to a responsive web interface where users can enter flight details and receive predictions in real time.

---

## ❓ Why I Built This

The goal was not simply to train a machine learning model and report its accuracy.

I wanted to build a complete **end-to-end ML application** that demonstrates how a machine learning model can move from raw data to a usable production-style application.

The project covers the complete workflow:

```text
Raw Flight Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train / Validation / Test Split
       ↓
LightGBM Model
       ↓
Model Evaluation
       ↓
Threshold Optimization
       ↓
Saved Model
       ↓
FastAPI REST API
       ↓
Interactive Web Dashboard
       ↓
Real-Time Prediction
This approach allowed me to work on both the data science and machine learning engineering sides of the project.
_ _ _

## 🧠 Machine Learning Approach

The core prediction model is **LightGBM (Light Gradient Boosting Machine)**.

LightGBM was selected because it is well suited for structured/tabular datasets and can efficiently learn complex relationships between flight characteristics and delay outcomes.

### Main Features

The model uses features such as:

- Route
- Airline
- Origin airport
- Destination airport
- Departure time
- Arrival time
- Day
- Day of week
- Month
- Distance
- Scheduled departure time
- Scheduled arrival time
- Elapsed flight time

Additional time-based features were engineered from the original flight information.

---

## 🔧 Feature Engineering

Instead of relying only on the original dataset columns, additional predictive features were created.

Examples include:

- `DepHour`
- `DepMinute`
- `ArrHour`
- `ArrMinute`
- `DayOfWeek`
- `Month`
- `CRSDepTime`
- `CRSArrTime`
- `CRSElapsedTime`
- `DepMinutes`
- `ArrMinutes`
- `Route`

These features help the model capture temporal, scheduling, and route-related patterns that may influence flight delays.

---

## 🎚️ Classification Threshold Optimization

A standard binary classifier commonly uses a probability threshold of `0.50`.

Instead of automatically using `0.50`, I evaluated multiple classification thresholds to understand how changing the threshold affects:

- Precision
- Recall
- F1 Score

The analysis identified an optimized classification threshold of:

```text
0.38
This threshold was then integrated into the prediction pipeline.
The purpose of threshold optimization was to select a decision boundary based on the model's precision-recall trade-off rather than relying on the default 0.50 threshold.
_ _ _

📊 Model Evaluation

The model was evaluated using multiple metrics and visualizations rather than relying only on accuracy.

The evaluation pipeline includes:

Precision
Recall
F1 Score
Confusion Matrix
Classification Threshold Analysis
Feature Importance
Prediction Probability

This provides a more complete understanding of how the model performs when identifying delayed and on-time flights.

🔍 Feature Importance

LightGBM feature importance was analyzed to understand which input variables contributed most strongly to the model.

The top features identified during the analysis included:

Route
Day
Origin
Destination
Scheduled departure time
Day of week
Scheduled arrival time
Arrival minute
Departure minute
Scheduled elapsed time

The feature importance analysis provides additional insight into the patterns learned by the model.

🚀 What I Added

This project was developed beyond a basic machine learning training notebook.

I converted the trained model into an end-to-end prediction application.

Machine Learning -
Data preprocessing pipeline
Feature engineering
LightGBM model training
Model evaluation
Feature importance analysis
Classification threshold optimization
Model serialization
Prediction pipeline

Backend -
FastAPI REST API
/health endpoint
/predict endpoint
Pydantic request validation
Real-time prediction service
JSON API responses
Automatic API documentation through Swagger/OpenAPI

Frontend -
Modern aviation-themed interface
Responsive dashboard
Flight input form
Real-time prediction results
Delay probability visualization
Risk indicator
Model status
API status
Route and airline information
JavaScript API integration

⚙️ Technologies Used
Programming Languages -
Python
JavaScript
HTML
CSS

Machine Learning -
LightGBM
Scikit-learn
Pandas
NumPy
Joblib
Backend
FastAPI
Uvicorn
Pydantic

Data Processing & Visualization -
Pandas
NumPy
Matplotlib
CSV

Development Tools -
Visual Studio Code
Git
GitHub
Python Virtual Environment

🔌 REST API

The trained machine learning model is exposed through a FastAPI REST API.
Health Check -
GET /health

The endpoint is used to verify that the API service is running correctly.
Prediction Endpoint -
POST /predict

The endpoint accepts flight information as JSON and returns the model prediction.
Example Request -
{
  "FlightDate": "2026-07-15",
  "Reporting_Airline": "DL",
  "Origin": "DTW",
  "Dest": "MKE",
  "CRSDepTime": 1649,
  "CRSArrTime": 1750,
  "CRSElapsedTime": 72,
  "Distance": 237
}

Example Response -
{
  "prediction": 1,
  "status": "Delayed",
  "delay_probability": 0.4284,
  "threshold": 0.38
}
The API returns both the prediction and the underlying delay probability, making the result more informative than a simple binary classification.

🖥️ Interactive Web Dashboard

The frontend provides a user-friendly interface for interacting with the prediction API.
Users can enter flight information and receive a prediction without directly interacting with Python or the backend API.

The dashboard displays:

Flight details
Predicted flight status
Delay probability
Classification threshold
Machine learning model
API status
Route
Airline information

The frontend communicates with the FastAPI backend using JavaScript fetch() requests.
<img width="1297" height="676" alt="image" src="https://github.com/user-attachments/assets/c1eea587-d5e2-4014-b5f4-328f4614cb8e" />

🏗️ System Architecture
                    ┌─────────────────────┐
                    │    Flight Dataset   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LightGBM       │
                    │   Model Training    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Threshold Analysis  │
                    │       0.38          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Saved ML Model    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      REST API       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Interactive Web UI  │
                    └─────────────────────┘

📁 Project Structure
flight-delay-predictor/
│
├── app/
│   └── api.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│   ├── flight_delay_lgbm.pkl
│   ├── threshold.json
│   ├── evaluation_results.csv
│   ├── threshold_results.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── threshold_analysis.png
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── prepare_model_data.py
│   ├── split_data.py
│   └── train_lgbm.py
│
├── tests/
│
├── .gitignore
└── README.md

▶️ How to Run Locally
1. Clone the Repository
git clone https://github.com/kamaal099/flight-delay-predictor.git
cd flight-delay-predictor

2. Create a Virtual Environment
python -m venv .venv

3. Activate the Environment
Windows PowerShell:
.venv\Scripts\Activate.ps1

4. Install Dependencies
pip install fastapi uvicorn lightgbm pandas numpy scikit-learn joblib

5. Start the API
uvicorn app.api:app --reload

The API will be available at:
http://127.0.0.1:8000

6. Open the API Documentation
FastAPI automatically provides interactive Swagger documentation:
http://127.0.0.1:8000/docs

🧪 Example Prediction
Example flight:
Airline: DL
Origin: DTW
Destination: MKE
Departure Time: 16:49
Arrival Time: 17:50
Distance: 237 miles

The model returns:
Prediction: Delayed
Delay Probability: 42.84%
Threshold: 38%

Since the predicted probability is above the optimized classification threshold of 0.38, the system classifies the flight as:
Delayed

📈 Evaluation & Analysis

The repository contains generated model analysis artifacts.
Confusion Matrix
The confusion matrix shows the number of:
True Positives
True Negatives
False Positives
False Negatives
Threshold Analysis

The threshold analysis visualizes how:
Precision
Recall
F1 Score

change as the classification threshold changes.

Feature Importance
The feature importance visualization shows which features contributed most strongly to the LightGBM model.

🔮 Future Improvements

Possible future improvements include:
Real-time weather data integration
Live flight status API integration
Airport congestion information
Airline historical performance features
SHAP-based model explainability
Automated model retraining
Model monitoring
Docker containerization
Cloud deployment
CI/CD pipeline
Database integration
Authentication and API keys
Production logging and monitoring

🎓 Key Learning Outcomes

This project provided practical experience across the complete machine learning lifecycle:
Data
  ↓
Data Cleaning
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Threshold Optimization
  ↓
Model Serialization
  ↓
REST API Development
  ↓
Frontend Integration
  ↓
End-to-End ML Application

The main objective was to demonstrate how a trained machine learning model can be transformed into a usable application rather than stopping at model training and evaluation.

👨‍💻 Author
Kamaal Ahmad
Computer Science | Data Science
GitHub: @kamaal099

⭐ Project
If you find this project useful, consider giving the repository a star.
**Important:** Paste this starting exactly at `## 🧠 Machine Learning Approach` after the content you already have.
