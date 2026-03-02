from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    # Katalog z danymi surowymi i wyczyszczonymi.
    data_dir: str = "data"
    # Katalog z modelami i plikami po drodze.
    artifacts_dir: str = "artifacts"
    # Katalog z wynikami metryk i benchmarku.
    outputs_dir: str = "outputs"
    # Katalog z tokenized danymi teachera.
    tokenized_teacher_dir: str = "artifacts/tokenized_teacher"
    # Katalog z tokenized danymi studenta.
    tokenized_student_dir: str = "artifacts/tokenized_student"


@dataclass
class DataConfig:
    # Zrodlo danych: URL lub lokalny plik CSV.
    source: str = "https://raw.githubusercontent.com/isaaccs/sentiment-analysis-for-financial-news/master/all-data.csv"
    # Kodowanie danych wejscia.
    encoding: str = "latin-1"
    # Separator, None oznacza auto logike buildera.
    separator: str | None = None
    # Nazwa kolumny tekstu, jesli znana.
    text_column: str | None = None
    # Nazwa kolumny etykiety, jesli znana.
    label_column: str | None = None
    # Mapa etykiet tekstowych do int.
    label_map: dict[str, int] = field(
        default_factory=lambda: {"negative": 0, "neutral": 1, "positive": 2}
    )
    # Seed splitu train/val/test.
    split_seed: int = 42
    # Proporcja test splitu pierwszego etapu.
    test_size: float = 0.2
    # Maksymalna dlugosc tokenizacji.
    max_length: int = 128

    def __post_init__(self) -> None:
        # Wymuszamy typy numeryczne, bo YAML potrafi zwrocic stringi.
        self.split_seed = int(self.split_seed)
        self.test_size = float(self.test_size)
        self.max_length = int(self.max_length)


@dataclass
class ModelConfig:
    # Nazwa modelu teachera.
    teacher_model_name: str = "distilroberta-base"
    # Nazwa modelu studenta.
    student_model_name: str = "prajjwal1/bert-tiny"
    # Liczba klas.
    num_labels: int = 3

    def __post_init__(self) -> None:
        # Pilnujemy typu liczby klas.
        self.num_labels = int(self.num_labels)


@dataclass
class TrainingConfig:
    # Batch size dla treningu i ewaluacji.
    batch_size: int = 8
    # Epoki baseline.
    baseline_epochs: int = 3
    # Epoki teachera.
    teacher_epochs: int = 3
    # Epoki distillation.
    distill_epochs: int = 5
    # Learning rate baseline i distillation.
    student_lr: float = 2e-5
    # Learning rate teachera.
    teacher_lr: float = 1e-5
    # Maksymalna norma gradientu.
    grad_clip: float = 1.0
    # Seed dla powtarzalnosci.
    seed: int = 42

    def __post_init__(self) -> None:
        # Konwertujemy wartosci do typow wymaganych przez PyTorch.
        self.batch_size = int(self.batch_size)
        self.baseline_epochs = int(self.baseline_epochs)
        self.teacher_epochs = int(self.teacher_epochs)
        self.distill_epochs = int(self.distill_epochs)
        self.student_lr = float(self.student_lr)
        self.teacher_lr = float(self.teacher_lr)
        self.grad_clip = float(self.grad_clip)
        self.seed = int(self.seed)


@dataclass
class DistillationConfig:
    # Temperature dla miekkich etykiet.
    temperature: float = 1.0
    # Balans CE i KL.
    alpha: float = 0.8
    # Czy zaczynac KD od baseline checkpointu.
    use_warm_start: bool = True

    def __post_init__(self) -> None:
        # Wymuszamy typy numeryczne i bool dla stabilnej konfiguracji.
        self.temperature = float(self.temperature)
        self.alpha = float(self.alpha)
        self.use_warm_start = bool(self.use_warm_start)


@dataclass
class BenchmarkConfig:
    # Liczba krokow warmup.
    warmup_steps: int = 50
    # Liczba krokow pomiarowych.
    measure_steps: int = 200
    # Batch sizes do benchmarku.
    batches: list[int] = field(default_factory=lambda: [1, 8])

    def __post_init__(self) -> None:
        # Pilnujemy, aby listy batchy byly intami.
        self.warmup_steps = int(self.warmup_steps)
        self.measure_steps = int(self.measure_steps)
        self.batches = [int(x) for x in self.batches]


@dataclass
class ProjectConfig:
    # Sekcja sciezek.
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Sekcja danych.
    data: DataConfig = field(default_factory=DataConfig)
    # Sekcja modeli.
    models: ModelConfig = field(default_factory=ModelConfig)
    # Sekcja treningu.
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Sekcja distillation.
    distill: DistillationConfig = field(default_factory=DistillationConfig)
    # Sekcja benchmarku.
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        # Ladujemy konfiguracje YAML i mapujemy na dataclasses.
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            paths=PathsConfig(**data.get("paths", {})),
            data=DataConfig(**data.get("data", {})),
            models=ModelConfig(**data.get("models", {})),
            training=TrainingConfig(**data.get("training", {})),
            distill=DistillationConfig(**data.get("distill", {})),
            benchmark=BenchmarkConfig(**data.get("benchmark", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        # Zapisujemy konfiguracje do YAML, aby latwo wersjonowac parametry.
        payload: dict[str, Any] = asdict(self)
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
