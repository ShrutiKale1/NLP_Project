from predict_emotion import predict_emotion
from google import genai

client = genai.Client(api_key="API_KEY")

user_text = input("Enter your question: ")

emotion_data = predict_emotion(user_text)

emotion = emotion_data["emotion"]
confidence = emotion_data["confidence"]

print("Detected Emotion:", emotion)
print("Confidence:", round(confidence,3))

if emotion == "Confused":
    instruction = "Explain the concept very simply step by step."

elif emotion == "Frustrated":
    instruction = "Explain calmly and encourage the student."

elif emotion == "Confident":
    instruction = "Give a deeper explanation and challenge the student."

print("Teaching Strategy:", instruction)

final_prompt = instruction + "\n\nStudent question: " + user_text

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=final_prompt
)

print("\nAI Tutor Response:\n")
print(response.text)