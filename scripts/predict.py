import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "../models/"
MODEL_NAME = "distilbert-base-uncased"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text, model, tokenizer, device="cuda", threshold=0.5):
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs)
        probs = torch.sigmoid(output.logits)  # Apply sigmoid activation

    predicted_labels = (probs > threshold).int().cpu().numpy().flatten()  # Convert to binary (0 or 1)

    return predicted_labels

if __name__ == "__main__":
    sample_text = "AI is revolutionizing healthcare."
    predicted_labels = predict(sample_text, model, tokenizer)
    print(f"Predicted labels: {predicted_labels}")