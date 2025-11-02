import streamlit as st
import requests


FASTAPI_URL = "http://127.0.0.1:8000"  # change to your deployed URL later

st.set_page_config(page_title="Student Stress Detection & Counselor", page_icon="💬", layout="centered")

st.title(" Student Stress Detection & AI Counselor")
st.markdown("Analyze your stress level and talk to an AI counselor who understands you.")

# =========================
# 🧾 Questionnaire Section
# =========================
with st.form("stress_form"):
    st.subheader("Student Wellbeing Questionnaire")

    self_esteem = st.slider("Self Esteem (1 = very low, 10 = very high)", 1, 10, 5)
    sleep_quality = st.slider("Sleep Quality (1 = poor, 10 = excellent)", 1, 10, 6)
    screen_time = st.slider("Daily Screen Time (hours)", 0.0, 15.0, 6.0)
    study_load = st.slider("Study Load (1 = light, 10 = heavy)", 1, 10, 5)
    gender = st.selectbox("Gender", ["Male", "Female"])
    physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

    submitted = st.form_submit_button("Predict Stress Level")

# =========================
# 🎯 Predict Stress Level
# =========================
if submitted:
    gender_encoded = 0 if gender == "Male" else 1
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}

    payload = {
        "self_esteem": self_esteem,
        "sleep_quality": sleep_quality,
        "Hours_of_Screen_Time": screen_time,
        "study_load": study_load,
        "Gender_encoded": gender_encoded,
        "Physical_Activity_encoded": activity_map[physical_activity]
    }

    with st.spinner("Analyzing stress level..."):
        try:
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
            if response.status_code == 200:
                stress_level = response.json()["stress_level"]
                st.session_state["stress_level"] = stress_level
                st.success(f" Predicted Stress Level: **{stress_level}**")
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# =========================
# 💬 Chat Interface
# =========================
st.divider()
st.subheader(" AI Counselor")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display previous messages
for chat in st.session_state["chat_history"]:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**AI Counselor:** {chat['ai']}")

user_message = st.text_input("Type your message...")

if st.button("Send") and user_message.strip():
    gender_val = gender
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    activity_val = activity_map[physical_activity]

    payload = {
        "message": user_message,
        "self_esteem": self_esteem,
        "sleep_quality": sleep_quality,
        "screen_time": screen_time,
        "study_load": study_load,
        "gender": gender_val,
        "physical_activity": activity_val
    }

    try:
        response = requests.post(f"{FASTAPI_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            reply = data["reply"]
            stress_level = data.get("stress_level", "unknown")

            st.session_state["chat_history"].append({
                "user": user_message,
                "ai": reply,
                "stress": stress_level
            })
            st.rerun()
        else:
            st.error(f"Chat failed: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
