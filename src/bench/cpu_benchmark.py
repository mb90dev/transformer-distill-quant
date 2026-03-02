from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk


@dataclass
class CPUBenchmarkRunner:
    # Warmup kroki przed pomiarem.
    warmup_steps: int = 50
    # Kroki pomiarowe.
    measure_steps: int = 200
    # Batch sizes do porownania.
    batches: tuple[int, ...] = (1, 8)
    # Urzadzenie benchmarku.
    device: torch.device = torch.device("cpu")

    def run_single(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        # Benchmark pojedynczego modelu dla danego wejscia.
        model.eval()
        model.to(self.device)

        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = model(input_ids=input_ids, attention_mask=attention_mask).logits

        times_ms: list[float] = []
        with torch.no_grad():
            for _ in range(self.measure_steps):
                t0 = time.perf_counter()
                _ = model(input_ids=input_ids, attention_mask=attention_mask).logits
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        p50 = float(np.percentile(times_ms, 50))
        p95 = float(np.percentile(times_ms, 95))
        throughput = float((input_ids.shape[0] * self.measure_steps) / (sum(times_ms) / 1000.0))
        return {
            "p50_ms": p50,
            "p95_ms": p95,
            "throughput_req_s": throughput,
        }

    def build_sample_batch_fn(self, tokenized_student_dir: str | Path):
        # Tworzymy funkcje budujaca batch z jednej probki testowej.
        ds = load_from_disk(str(tokenized_student_dir))
        sample = ds["test"][0]

        def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
            input_ids = sample["input_ids"].unsqueeze(0).repeat(batch_size, 1).to(self.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).repeat(batch_size, 1).to(self.device)
            return input_ids, attention_mask

        return make_batch

    def run_suite(self, models: dict[str, torch.nn.Module], make_batch_fn) -> dict[str, dict[str, float]]:
        # Uruchamiamy benchmark dla wszystkich modeli i batch sizes.
        results: dict[str, dict[str, float]] = {}
        for bs in self.batches:
            input_ids, attention_mask = make_batch_fn(bs)
            for model_name, model in models.items():
                key = f"{model_name}_bs{bs}"
                results[key] = self.run_single(model, input_ids, attention_mask)
        return results

    @staticmethod
    def save_results(results: dict[str, dict[str, float]], output_json: str | Path) -> None:
        # Zapisujemy wyniki benchmarku do JSON.
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
