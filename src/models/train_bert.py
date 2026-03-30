import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

from src.utils.config import PROCESSED_DATA_PATH

MODEL_SAVE_PATH = "saved_models/bert/"

def train_bert():
    print("Loading data...")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 🔥 TAKE SMALL SAMPLE (IMPORTANT)
    df = df.sample(10000, random_state=42)

    X = df["clean_text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=100,
)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training BERT...")
    trainer.train()

    print("Evaluating BERT...")
    results = trainer.evaluate()

    print(results)

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print("BERT model saved successfully.")

if __name__ == "__main__":
    train_bert()