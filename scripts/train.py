import torch
from transformers import AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from dataset_loader import get_datasets

MODEL_NAME = "distilbert-base-uncased"

# Load datasets
tokenized_datasets = get_datasets()
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=16)

# Load model (change num_labels based on the number of possible categories)
num_labels = 5  # Adjust based on your dataset
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification")
optimizer = AdamW(model.parameters(), lr=5e-5)

def train(model, train_dataloader, optimizer, device="cuda"):
    model.train()
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()  # Use Binary Cross-Entropy loss for multi-label classification

    for batch in train_dataloader:
        inputs = {key: torch.tensor(val).to(device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
        labels = batch["label"].to(device).float()  # Ensure labels are floats for BCEWithLogitsLoss

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Training Loss: {loss.item()}")

if __name__ == "__main__":
    train(model, train_dataloader, optimizer)
    model.save_pretrained("../models/text_classifier")