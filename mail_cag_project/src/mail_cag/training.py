from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class TrainResult:
    model_dir: Path
    eval_accuracy: float


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def train_albert(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    max_length: int,
    learning_rate: float,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_labels: int,
    checkpoint_dir: Path | None = None,
) -> TrainResult:
    """Fine-tune one ALBERT classifier and save it."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_name)
    model = load_classifier(model_name, num_labels).to(device)

    train_loader = DataLoader(
        EmailDataset(train_df, tokenizer, max_length),
        batch_size=train_batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        EmailDataset(eval_df, tokenizer, max_length),
        batch_size=eval_batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        average_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch {epoch}/{epochs} loss: {average_loss:.4f}", flush=True)
        if checkpoint_dir is not None:
            save_current_checkpoint(model, tokenizer, checkpoint_dir, epoch)

    accuracy = evaluate_accuracy(model, eval_loader, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return TrainResult(model_dir=output_dir, eval_accuracy=accuracy)


def save_current_checkpoint(model, tokenizer, checkpoint_dir: Path, epoch: int) -> None:
    """Overwrite the current-round checkpoint after each finished epoch."""

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "epoch.txt").write_text(f"{epoch}\n", encoding="utf-8")


def evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating", leave=False):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(**batch).logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / total if total else 0.0


def add_true_label_confidence(
    *,
    df: pd.DataFrame,
    model_dir: Path,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    """Score each row by the model confidence for its true label."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)
    loader = DataLoader(EmailDataset(df, tokenizer, max_length), batch_size=batch_size)

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="scoring emails", leave=False):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            probs = model(**batch).logits.softmax(dim=1)
            scores.extend(probs[torch.arange(len(labels)), labels].cpu().tolist())

    result = df.copy()
    result["true_label_confidence"] = scores
    return result


def evaluate_saved_model(
    *,
    model_dir: Path,
    eval_df: pd.DataFrame,
    max_length: int,
    batch_size: int,
) -> dict[str, float | int]:
    """Evaluate one saved model on a clean dataframe."""

    labels, predictions = predict_saved_model(
        model_dir=model_dir,
        eval_df=eval_df,
        max_length=max_length,
        batch_size=batch_size,
    )
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def predict_saved_model(
    *,
    model_dir: Path,
    eval_df: pd.DataFrame,
    max_length: int,
    batch_size: int,
) -> tuple[list[int], list[int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)
    loader = DataLoader(EmailDataset(eval_df, tokenizer, max_length), batch_size=batch_size)

    labels: list[int] = []
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating saved model", leave=False):
            batch_labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_predictions = model(**batch).logits.argmax(dim=1)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
    return labels, predictions


def load_tokenizer(model_name_or_dir: str | Path):
    """Load tokenizers from the local Hugging Face cache only."""

    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_dir,
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not load tokenizer locally: {model_name_or_dir}. "
            "Run `scripts/download_models.sh` while online, then try again."
        ) from exc


def load_classifier(model_name: str, num_labels: int):
    """Load ALBERT locally and create the task-specific classifier head."""

    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not load model locally: {model_name}. "
            "Run `scripts/download_models.sh` while online, then try again."
        ) from exc
