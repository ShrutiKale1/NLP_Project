from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained("emotion_model")
model = BertForSequenceClassification.from_pretrained("emotion_model")

# Set model to evaluation mode
model.eval()

# Emotion labels
labels = ["Confused", "Frustrated", "Confident"]

def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()

    return {
        "emotion": labels[predicted_class],
        "confidence": float(probs[0][predicted_class])
    }


# Example test
if __name__ == "__main__":
    text = "I am really confused about this topic"
    result = predict_emotion(text)
    print(result)