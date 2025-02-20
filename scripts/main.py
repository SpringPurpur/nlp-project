import argparse
import os
import json
from train import train
from evaluate import evaluate
from predict import predict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# List of models to store locally
MODEL_NAMES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base"
]

SAVE_DIR = os.path.join(os.path.dirname(__file__), "../models")  # Directory where models are saved

# Define label names (update based on your dataset)
LABEL_NAMES = ["AI", "Healthcare", "Agriculture", "Education", "Finance"]

def is_model_downloaded(model_name):
    """Check if the model already exists locally."""
    model_path = os.path.join(SAVE_DIR, model_name)
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    return all(os.path.exists(os.path.join(model_path, file)) for file in required_files)

def download_models():
    """Download and save models locally if not already present, with user confirmation."""
    missing_models = [model for model in MODEL_NAMES if not is_model_downloaded(model)]

    if not missing_models:
        print("All required models are already downloaded. Skipping download.")
        return

    print("The following models are missing and need to be downloaded:")
    for model in missing_models:
        print(f"   - {model}")

    user_input = input("Do you want to download them now? (y/n): ").strip().lower()
    if user_input != "y":
        print("Missing models detected. Exiting...")
        exit(1)

    for model_name in missing_models:
        model_path = os.path.join(SAVE_DIR, model_name)
        print(f"Downloading {model_name}...")

        # Load the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save locally
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        print(f"{model_name} saved to {model_path}\n")

    print("All required models are now downloaded and ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-based Text Classifier")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "predict"])
    parser.add_argument("--text", type=str, help="Text to classify (required for predict mode)")

    args = parser.parse_args()

    print("Checking for required models...")
    download_models()

    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "predict":
        if args.text:
            predicted_labels = predict(args.text)

            # Convert binary outputs to human-readable labels
            readable_labels = [LABEL_NAMES[i] for i, val in enumerate(predicted_labels) if val == 1]

            print(f"Predicted categories: {', '.join(readable_labels) if readable_labels else 'None'}")
        else:
            print("Please provide text for classification.")
