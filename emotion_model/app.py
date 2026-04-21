import streamlit as st

# Gradient background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #ffdde1, #ee9ca7);
    }
    </style>
    """,
    unsafe_allow_html=True
)

import streamlit as st
from predict_emotion import predict_emotion
from google import genai

# Gemini client
client = genai.Client(api_key="API_KEY")

# Page config
st.set_page_config(page_title="Emotion-Adaptive AI Tutor", layout="centered")

# Title
st.title("Emotion-Adaptive AI Tutor")

# Input
user_text = st.text_area("Enter your question:")

# Button
if st.button("Generate Response"):

    if user_text.strip() == "":
        st.warning("Please enter a question")

    else:
        # ---------- Emotion Detection ----------
        emotion_data = predict_emotion(user_text)

        emotion = emotion_data["emotion"]
        confidence = emotion_data["confidence"]

        st.subheader("Detected Emotion")
        st.write(f"Emotion: {emotion}")
        st.write(f"Confidence: {round(confidence,3)}")

        # ---------- Teaching Strategy ----------
        if emotion == "Confused":
            instruction = "Explain the concept very simply step by step."

        elif emotion == "Frustrated":
            instruction = "Explain calmly and encourage the student."

        elif emotion == "Confident":
            instruction = "Give a deeper explanation and challenge the student."

        else:
            instruction = "Explain clearly."

        st.subheader("Teaching Strategy")
        st.info(instruction)

        # ---------- Gemini Response ----------
        final_prompt = instruction + "\n\nStudent question: " + user_text

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt
        )

        # ---------- Output ----------
        st.subheader("AI Tutor Response")
        st.success(response.text)