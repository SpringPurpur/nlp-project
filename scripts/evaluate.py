import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset_loader import get_datasets
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import multilabel_confusion_matrix, classification_report

MODEL_PATH = "../models/text_classifier"

def evaluate(model, eval_dataloader, device="cuda", threshold=0.5):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {key: torch.tensor(val).to(device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            predictions = (probs > threshold).int()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate multi-label confusion matrices
    cm = multilabel_confusion_matrix(all_labels, all_preds)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    # Plot confusion matrices per label
    num_labels = len(cm)
    fig, axes = plt.subplots(1, num_labels, figsize=(num_labels * 4, 4))

    for i in range(num_labels):
        sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"Label {i}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.show()

if __name__ == "__main__":
    tokenized_datasets = get_datasets()
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=16)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    evaluate(model, eval_dataloader)
