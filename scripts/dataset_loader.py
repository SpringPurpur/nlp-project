import json
import re
from datasets import Dataset

LABEL_NAMES = ["healthcare", "finance", "education", "sports", "technology"]  # Define your categories

def clean_text(text):
    """Remove URLs, extra spaces, and filter out unwanted content."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"‘|’|“|”", "'", text)  # Normalize quotes
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def load_json_dataset(file_path):
    """Load and preprocess dataset from JSON."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = [clean_text(entry["text"]) for entry in data]  # Clean article text
    labels = [[1 if label in entry["labels"] else 0 for label in LABEL_NAMES] for entry in data]  # Multi-label encoding

    return Dataset.from_dict({"text": texts, "labels": labels})

# Example usage
if __name__ == "__main__":
    dataset = load_json_dataset("data/articles.json")
    print(dataset)
