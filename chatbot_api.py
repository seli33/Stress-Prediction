from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- Load environment ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Google API key not found")

# ---------------- FastAPI App ----------------
app = FastAPI(title="AI Chatbot")

# ---------------- Enable CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict in production)
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Input Schema ----------------
class ChatRequest(BaseModel):
    message: str
    stress_level: Optional[str] = "unknown"

# ---------------- Initialize LLM ----------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# ---------------- Helper Functions ----------------
def detect_crisis(message: str) -> bool:
    CRISIS_KEYWORDS = ["suicide", "kill myself", "end it", "hopeless", "can't go on"]
    return any(k in message.lower() for k in CRISIS_KEYWORDS)

def get_therapist_response(stress_level: str, message: str):
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

    prompt = f"""
    You are a friendly AI counselor for students.
    The student's stress level is {stress_level}.
    Use a {tone} tone.
    They said: "{message}"

    Respond with short, empathetic advice,
    give 2–3 coping tips,
    and end with one open question to continue.
    Add this note at the end:
    "Note: I'm not a licensed therapist, but I can offer some general guidance."
    """
    response = llm.invoke(prompt)
    return response.content.strip()

# ---------------- Chatbot Endpoint ----------------
@app.post("/chatbot")
async def chatbot_endpoint(req: ChatRequest):
    try:
        reply = get_therapist_response(req.stress_level, req.message)
        return {"stress_level": req.stress_level, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
