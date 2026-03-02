from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    # Ustawiamy seed we wszystkich glownych generatorach losowosci.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cpu_device() -> torch.device:
    # Projekt zaklada CPU-only, wiec zwracamy CPU.
    return torch.device("cpu")
