from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_classifier(model: torch.nn.Module, dataloader: Any, device: torch.device) -> dict[str, float]:
    # Mierzymy accuracy i F1 macro bez liczenia gradientow.
    model.eval()
    preds: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(y.cpu().tolist())

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }
