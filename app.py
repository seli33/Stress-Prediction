from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib, os
from datetime import datetime
from typing import Optional
import pickle
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# === Load environment and models ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Google API key not found")

scaler = joblib.load("scaler.pkl")
stress_model = joblib.load("xgb_stress_model.pkl")

CSV_FILE = "stress_log.csv"
stress_map = {0: "Normal", 1: "Stressed", 2: "Highly Prone"}


app = FastAPI(title="Student Stress Detection & Chat Agent")

# === Input Schemas ===
class Questionnaire(BaseModel):
    self_esteem: float
    sleep_quality: float
    Hours_of_Screen_Time: float
    study_load: float
    Gender_encoded: int
    Physical_Activity_encoded: float

class ChatRequest(BaseModel):
    message: str
    stress_level: Optional[str] = "unknown"

# === Initialize LLM ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# === Helper functions ===
def detect_crisis(message: str) -> bool:
    CRISIS_KEYWORDS = ["suicide", "kill myself", "end it", "hopeless", "can't go on"]
    return any(k in message.lower() for k in CRISIS_KEYWORDS)

def get_ai_response(stress_level: str, message: str, features: dict = None) -> str:
    if detect_crisis(message):
        return (
            "I'm really sorry you're feeling this way. You're not alone — "
            "please reach out to someone you trust or contact local emergency services. "
            "If you’re in immediate danger, call your local emergency number."
        )

    tone = {
        "Normal": "positive and encouraging",
        "Stressed": "reassuring and supportive",
        "Highly Prone": "calm and empathetic",
        "unknown": "neutral and understanding"
    }.get(stress_level, "neutral")
    
    feature_info = ""
    if features:
        feature_info = "\nConsider these factors in your daily life:\n"
        for k, v in features.items():
            feature_info += f"- {k.replace('_', ' ').capitalize()}: {v}\n"

    prompt = f"""
    You are a friendly AI counselor for students.
    The student's stress level is {stress_level}.
    Use a {tone} tone.
    They said: "{message}"

    {feature_info}

    Explain why they might feel this way, provide short advice, 2–3 coping tips,
    and end with one open question to continue.
    Note: I'm not a licensed therapist, but I can offer some general guidance.
    """
    response = llm.invoke(prompt)
    return response.content.strip()


# === API Endpoints ===
@app.post("/predict")
def predict_stress(data: Questionnaire):
    try:
        df = pd.DataFrame([data.model_dump()])
        scaled = scaler.transform(df)

        # Convert prediction to plain Python type
        pred_num = int(stress_model.predict(scaled)[0])

        # Map numeric labels to string manually
        label_map = {0: "Normal", 1: "Stressed", 2: "Highly prone"}
        stress_level = label_map.get(pred_num, "unknown")


   
        # Prepare record to save
        today = pd.to_datetime(datetime.now().date())
        record = pd.DataFrame({
            "date": [today],
            "day": [datetime.now().strftime("%a")],
            "prediction_value": [pred_num],
            "prediction_label": [stress_level]
        })

        # Save to CSV
        if os.path.exists(CSV_FILE)and os.path.getsize(CSV_FILE) > 0:
            df_existing = pd.read_csv(CSV_FILE)
            if 'date' in df_existing.columns:
                df_existing['date'] = pd.to_datetime(df_existing['date']).dt.normalize()
            df_combined = pd.concat([df_existing, record], ignore_index=True)
            df_combined.to_csv(CSV_FILE, index=False)
        else:
            record.to_csv(CSV_FILE, index=False)

        return {"stress_level": stress_level, "prediction_value": pred_num}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ---------------- Weekly Summary ----------------
@app.get("/weekly_summary")
async def weekly_summary():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return {"error": "No data found."}

    # Load CSV
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return {"error": "No data available."}

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # Filter last 7 days
    last_week = datetime.now().date() - pd.Timedelta(days=7)
    week_data = df[df['date'] > pd.to_datetime(last_week)]

    if week_data.empty:
        return {"error": "No data for the past week."}

    # Weekly overall: mode of prediction_value
    overall_value = int(week_data['prediction_value'].mode()[0])
    overall_label = stress_map.get(overall_value, "unknown")

    # Daily list
    daily_list = week_data[['date', 'day', 'prediction_value', 'prediction_label']]\
        .to_dict(orient='records')

    return {
        "daily": daily_list,
        "weekly_overall": {
            "prediction_value": overall_value,
            "prediction_label": overall_label
        }
    }

@app.post("/chat")
def chat_endpoint(data: ChatRequest):
    try:
        # Optional: pass features from prediction to help AI explain stress level
        features = {
            "self_esteem": data.self_esteem if hasattr(data, "self_esteem") else "unknown",
            "sleep_quality": data.sleep_quality if hasattr(data, "sleep_quality") else "unknown",
            "Hours_of_Screen_Time": data.Hours_of_Screen_Time if hasattr(data, "Hours_of_Screen_Time") else "unknown",
            "study_load": data.study_load if hasattr(data, "study_load") else "unknown",
            "Physical_Activity": data.Physical_Activity_encoded if hasattr(data, "Physical_Activity_encoded") else "unknown"
        }
        
        reply = get_ai_response(data.stress_level or "unknown", data.message, features)
        return {"stress_level": data.stress_level, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
