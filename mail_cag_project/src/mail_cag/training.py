from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
import json
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class TrainResult:
    model_dir: Path
    eval_accuracy: float
    eval_metrics: dict[str, float | int | list[list[int]]]


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


def train_transformer_classifier(
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
    """Fine-tune one Hugging Face sequence classifier and save it."""

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
    eval_history: list[dict[str, float | int | list[list[int]]]] = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
        )
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        average_loss = running_loss / max(len(train_loader), 1)
        tqdm.write(f"epoch {epoch}/{epochs} loss: {average_loss:.4f}")
        if checkpoint_dir is not None:
            save_current_checkpoint(model, tokenizer, checkpoint_dir, epoch)
        epoch_metrics = evaluate_model(model, eval_loader, device)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["train_loss"] = average_loss
        eval_history.append(epoch_metrics)
        write_training_metrics(output_dir.parent, eval_history, epoch_metrics)
        tqdm.write(
            "epoch "
            f"{epoch}/{epochs} eval: "
            f"accuracy={epoch_metrics['accuracy']:.4f}, "
            f"f1={epoch_metrics['f1']:.4f}, "
            f"benign_fpr={epoch_metrics['benign_false_positive_rate']:.4f}, "
            f"phishing_fnr={epoch_metrics['phishing_false_negative_rate']:.4f}, "
            f"pred_phishing={epoch_metrics['predicted_phishing_count']}/"
            f"{epoch_metrics['rows']}"
        )
        model.train()

    final_metrics = eval_history[-1] if eval_history else evaluate_model(model, eval_loader, device)
    accuracy = float(final_metrics["accuracy"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return TrainResult(
        model_dir=output_dir,
        eval_accuracy=accuracy,
        eval_metrics=final_metrics,
    )


def save_current_checkpoint(model, tokenizer, checkpoint_dir: Path, epoch: int) -> None:
    """Overwrite the current-round checkpoint after each finished epoch."""

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "epoch.txt").write_text(f"{epoch}\n", encoding="utf-8")


def evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    return float(evaluate_model(model, loader, device)["accuracy"])


def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int | list[list[int]]]:
    labels: list[int] = []
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="evaluating",
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
        ):
            batch_labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_predictions = model(**batch).logits.argmax(dim=1)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
    return classification_metrics(labels, predictions)


def classification_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float | int | list[list[int]]]:
    if not labels:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "benign_recall": 0.0,
            "phishing_recall": 0.0,
            "benign_false_positive_rate": 0.0,
            "phishing_false_negative_rate": 0.0,
            "predicted_benign_count": 0,
            "predicted_phishing_count": 0,
            "true_benign_count": 0,
            "true_phishing_count": 0,
        }

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["benign", "phishing"],
        zero_division=0,
        output_dict=True,
    )
    true_benign = int(tn + fp)
    true_phishing = int(fn + tp)
    predicted_benign = int(tn + fn)
    predicted_phishing = int(fp + tp)
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "benign_recall": float(report["benign"]["recall"]),
        "phishing_recall": float(report["phishing"]["recall"]),
        "benign_false_positive_rate": fp / true_benign if true_benign else 0.0,
        "phishing_false_negative_rate": fn / true_phishing if true_phishing else 0.0,
        "predicted_benign_count": predicted_benign,
        "predicted_phishing_count": predicted_phishing,
        "true_benign_count": true_benign,
        "true_phishing_count": true_phishing,
    }


def write_training_metrics(
    round_dir: Path,
    history: list[dict[str, float | int | list[list[int]]]],
    latest: dict[str, float | int | list[list[int]]],
) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    serializable_history = [
        {key: value for key, value in row.items() if key != "confusion_matrix"}
        for row in history
    ]
    pd.DataFrame(serializable_history).to_csv(
        round_dir / "training_eval_history.csv",
        index=False,
    )
    (round_dir / "training_eval_metrics.json").write_text(
        json.dumps(latest, indent=2),
        encoding="utf-8",
    )


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
        for batch in tqdm(
            loader,
            desc="scoring emails",
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
        ):
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
        for batch in tqdm(
            loader,
            desc="evaluating saved model",
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
        ):
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
    """Load a local Hugging Face classifier and create the task head."""

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
