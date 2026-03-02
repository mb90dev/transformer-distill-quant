from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DynamicQuantizer:
    # Typ docelowy wag quantized.
    dtype: torch.dtype = torch.qint8

    def quantize_linear(self, model_fp32: torch.nn.Module) -> torch.nn.Module:
        # Dynamic quantization: zamieniamy warstwy Linear na INT8.
        model_fp32.eval()
        return torch.quantization.quantize_dynamic(
            model_fp32,
            {torch.nn.Linear},
            dtype=self.dtype,
        )

    def save(
        self,
        model_int8: torch.nn.Module,
        output_dir: str | Path,
        prefix: str = "model_int8",
    ) -> tuple[Path, Path]:
        # Zapisujemy pelny obiekt i state_dict modelu INT8.
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        full_path = output / f"{prefix}_full.pt"
        state_path = output / f"{prefix}_state_dict.pt"

        torch.save(model_int8, full_path)
        torch.save(model_int8.state_dict(), state_path)
        return full_path, state_path

    def load_from_state_dict(self, fp32_model: torch.nn.Module, state_path: str | Path) -> torch.nn.Module:
        # Budujemy model INT8 z FP32 i ladujemy wagi z state_dict.
        model_int8 = self.quantize_linear(fp32_model)
        state = torch.load(state_path, map_location="cpu")
        model_int8.load_state_dict(state)
        return model_int8
