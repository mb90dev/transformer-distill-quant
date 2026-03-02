from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..eval.metrics import evaluate_classifier


@dataclass
class TeacherWorkflow:
    # Nazwa modelu teachera.
    teacher_model_name: str
    # Liczba klas.
    num_labels: int
    # Urzadzenie treningu/inferencji.
    device: torch.device
    # LR teachera.
    lr: float = 1e-5
    # Epoki fine-tuningu teachera.
    epochs: int = 3
    # Clip gradientow.
    grad_clip: float = 1.0

    def load(self, model_path: str | Path | None = None) -> None:
        # Ladujemy teachera z lokalnego katalogu albo z hub.
        if model_path and Path(model_path).joinpath("config.json").exists():
            source = str(model_path)
        else:
            source = self.teacher_model_name

        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=self.num_labels,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        # Ewaluacja teachera na podanym dataloaderze.
        return evaluate_classifier(self.model, dataloader, self.device)

    def finetune_if_needed(
        self,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        target_test_f1: float = 0.6,
        enable_finetune: bool = True,
    ) -> dict[str, Any]:
        # Najpierw mierzymy teachera, potem opcjonalnie trenujemy jesli jest slaby.
        before_val = self.evaluate(val_loader)
        before_test = self.evaluate(test_loader)

        should_finetune = enable_finetune and (before_test["f1_macro"] < target_test_f1)

        if should_finetune:
            for epoch in range(self.epochs):
                self.model.train()
                pbar = tqdm(train_loader, desc=f"Teacher epoch {epoch + 1}/{self.epochs}")
                for batch in pbar:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    loss = self.loss_fn(logits, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    pbar.set_postfix({"loss": round(float(loss.detach().cpu()), 4)})

        after_val = self.evaluate(val_loader)
        after_test = self.evaluate(test_loader)

        return {
            "before": {"validation": before_val, "test": before_test},
            "after": {"validation": after_val, "test": after_test},
            "finetuned": bool(should_finetune),
        }

    def save_metrics(self, metrics: dict[str, Any], output_json: str | Path) -> None:
        # Zapisujemy metryki teachera.
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def save_model(self, output_dir: str | Path, safe_serialization: bool = False) -> None:
        # Zapisujemy finalna wersje teachera i tokenizera.
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output)

    def save_logits_for_splits(
        self,
        tokenized_teacher: DatasetDict,
        output_dir: str | Path,
        batch_size: int = 8,
    ) -> None:
        # Liczymy logits teachera dla train/validation/test i zapisujemy je do npz.
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for split in ["train", "validation", "test"]:
                loader = DataLoader(tokenized_teacher[split], batch_size=batch_size, shuffle=False)
                all_idx: list[int] = []
                all_logits: list[np.ndarray] = []
                all_labels: list[int] = []

                for batch in tqdm(loader, desc=f"Teacher logits: {split}"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

                    all_idx.extend(batch["idx"].cpu().tolist())
                    all_logits.append(logits.cpu().numpy())
                    all_labels.extend(batch["label"].cpu().tolist())

                logits_np = np.concatenate(all_logits, axis=0)
                np.savez(
                    out / f"teacher_logits_{split}.npz",
                    idx=np.array(all_idx),
                    logits=logits_np,
                    labels=np.array(all_labels),
                )
