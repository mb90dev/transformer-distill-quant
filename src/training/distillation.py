from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..eval.metrics import evaluate_classifier


@dataclass
class DistillationTrainer:
    # Zrodlo modelu studenta (pretrained lub path).
    student_model_name: str
    # Liczba klas.
    num_labels: int
    # Urzadzenie treningu.
    device: torch.device
    # LR studenta.
    lr: float = 2e-5
    # Gradient clipping.
    grad_clip: float = 1.0
    # Temperature KD.
    temperature: float = 1.0
    # Alpha CE/KL.
    alpha: float = 0.8
    # Czy korzystac z warm-start baseline.
    use_warm_start: bool = True

    def load_student(self, warm_start_path: str | Path | None = None) -> str:
        # Ladujemy studenta z warm-start jesli jest dostepny, inaczej z pretrained.
        source = self.student_model_name
        if self.use_warm_start and warm_start_path and Path(warm_start_path).joinpath("config.json").exists():
            source = str(warm_start_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=self.num_labels,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        self.tokenizer_source = self.student_model_name
        return source

    @staticmethod
    def build_teacher_map(logits_npz_path: str | Path) -> dict[int, np.ndarray]:
        # Budujemy mapowanie idx -> logits teachera.
        blob = np.load(logits_npz_path)
        idx = blob["idx"]
        logits = blob["logits"]
        return {int(i): logits[pos] for pos, i in enumerate(idx)}

    def distillation_loss_components(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Zwracamy osobno CE, KL i total loss.
        ce_loss = self.ce_loss_fn(student_logits, labels)

        if self.alpha >= 1.0:
            kd_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
            return ce_loss, kd_loss, ce_loss

        if teacher_logits_batch is None:
            raise ValueError("Brak teacher_logits_batch przy alpha < 1.0")

        log_p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_teacher = F.softmax(teacher_logits_batch / self.temperature, dim=-1)
        kd_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (self.temperature ** 2)
        total = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
        return ce_loss, kd_loss, total

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        teacher_map: dict[int, np.ndarray] | None,
        epochs: int,
        best_ckpt_dir: str | Path,
    ) -> dict[str, Any]:
        # Trenujemy studenta KD i zapisujemy najlepszy checkpoint po val f1.
        best_val_f1 = -1.0
        history: list[dict[str, float]] = []
        best_dir = Path(best_ckpt_dir)

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Distill epoch {epoch + 1}/{epochs}")
            sum_ce, sum_kd, sum_total, steps = 0.0, 0.0, 0.0, 0

            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                teacher_logits_batch = None
                if self.alpha < 1.0:
                    if teacher_map is None:
                        raise ValueError("teacher_map jest wymagany gdy alpha < 1.0")
                    idx = batch["idx"].cpu().tolist()
                    teacher_logits_batch = torch.tensor(
                        [teacher_map[int(i)] for i in idx],
                        dtype=torch.float32,
                        device=self.device,
                    )

                student_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                ce_loss, kd_loss, total_loss = self.distillation_loss_components(
                    student_logits,
                    labels,
                    teacher_logits_batch,
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                sum_ce += float(ce_loss.detach().cpu())
                sum_kd += float(kd_loss.detach().cpu())
                sum_total += float(total_loss.detach().cpu())
                steps += 1

                pbar.set_postfix(
                    {
                        "loss": round(float(total_loss.detach().cpu()), 4),
                        "ce": round(float(ce_loss.detach().cpu()), 4),
                        "kd": round(float(kd_loss.detach().cpu()), 4),
                    }
                )

            val_metrics = evaluate_classifier(self.model, val_loader, self.device)
            mean_ce = sum_ce / max(1, steps)
            mean_kd = sum_kd / max(1, steps)
            mean_total = sum_total / max(1, steps)

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "mean_ce_loss": mean_ce,
                    "mean_kd_loss": mean_kd,
                    "mean_total_loss": mean_total,
                }
            )

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(best_dir)
                AutoTokenizer.from_pretrained(self.tokenizer_source).save_pretrained(best_dir)

        if best_dir.joinpath("config.json").exists():
            self.model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(self.device)

        validation = evaluate_classifier(self.model, val_loader, self.device)
        test = evaluate_classifier(self.model, test_loader, self.device)
        return {
            "validation": validation,
            "test": test,
            "history": history,
            "best_checkpoint": str(best_dir),
        }

    def save(self, output_dir: str | Path) -> None:
        # Zapisujemy finalny model studenta KD.
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_source)
        self.model.save_pretrained(output)
        tokenizer.save_pretrained(output)

    @staticmethod
    def save_metrics(
        metrics: dict[str, Any],
        metrics_path: str | Path,
        history_path: str | Path | None = None,
    ) -> None:
        # Zapisujemy metryki i opcjonalnie historie epok.
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"validation": metrics["validation"], "test": metrics["test"]}, indent=2),
            encoding="utf-8",
        )

        if history_path is not None:
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(json.dumps(metrics.get("history", []), indent=2), encoding="utf-8")
