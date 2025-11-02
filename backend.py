from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
import os

app = FastAPI(title="Student Stress Prediction")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model, scaler
model = joblib.load("xgb_stress_model.pkl")
scaler = joblib.load("scaler.pkl")

file_path = "weekly_stress_log.csv"
stress_map = {0: "Normal", 1: "Stressed", 2: "Highly Prone"}

# ---------------- Input Schema ----------------
class StressInput(BaseModel):
    self_esteem: float
    sleep_quality: float
    Hours_of_Screen_Time: float
    study_load: float
    Gender_encoded: float
    Physical_Activity_encoded: float

# ---------------- Predict Endpoint ----------------
@app.post("/predict")
async def predict(data: StressInput):
    X_input = pd.DataFrame([data.dict()])
    scaled_X_input = scaler.transform(X_input)
    prediction = int(model.predict(scaled_X_input)[0])
    label = stress_map[prediction]

    today = pd.to_datetime(datetime.now().date())
    record = pd.DataFrame({
        "date": [today],
        "day": [datetime.now().strftime("%a")],
        "prediction_value": [prediction],
        "prediction_label": [label]
    })

    # Save record to CSV
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = pd.concat([df, record], ignore_index=True)
        df.to_csv(file_path, index=False)
    else:
        record.to_csv(file_path, index=False)

    response = {"prediction": prediction, "label": label}
    if prediction in [1, 2]:
        response["chatbot_offer"] = "You seem stressed. Would you like to chat with the AI assistant?"
    return response

# ---------------- Weekly Summary ----------------
@app.get("/weekly_summary")
async def weekly_summary():
    if not os.path.exists(file_path):
        return {"error": "No data found."}

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    last_week = datetime.now() - pd.Timedelta(days=7)
    week_data = df[df['date'] > last_week]

    if week_data.empty:
        return {"error": "No data for the past week."}

    overall_value = int(week_data['prediction_value'].mode()[0])
    overall_label = stress_map[overall_value]
    daily_list = week_data[['date','day','prediction_value','prediction_label']].to_dict(orient='records')

    return {"daily": daily_list, "weekly_overall": {"prediction_value": overall_value, "prediction_label": overall_label}}
