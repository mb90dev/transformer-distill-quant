from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..eval.metrics import evaluate_classifier


@dataclass
class BaselineTrainer:
    # Model studenta do treningu.
    model: torch.nn.Module
    # Nazwa zrodla tokenizera.
    tokenizer_source: str
    # Urzadzenie treningu.
    device: torch.device
    # Learning rate.
    lr: float = 2e-5
    # Maksymalna norma gradientu.
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        # Tworzymy optymalizator i CE loss raz po inicjalizacji.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def fit(self, train_loader: Any, val_loader: Any, test_loader: Any, epochs: int = 3) -> dict[str, Any]:
        # Trenujemy model baseline CE i zwracamy metryki + historie strat.
        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Baseline epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            steps = 0

            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = self.ce_loss(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += float(loss.detach().cpu())
                steps += 1
                pbar.set_postfix({"loss": round(float(loss.detach().cpu()), 4)})

            val_metrics = evaluate_classifier(self.model, val_loader, self.device)
            mean_loss = epoch_loss / max(1, steps)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_train_loss": mean_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                }
            )

        validation = evaluate_classifier(self.model, val_loader, self.device)
        test = evaluate_classifier(self.model, test_loader, self.device)
        return {"validation": validation, "test": test, "history": history}

    def save(self, output_dir: str | Path) -> None:
        # Zapisujemy model i tokenizer baseline.
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_source)
        self.model.save_pretrained(output)
        tokenizer.save_pretrained(output)
