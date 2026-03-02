"""Moduly treningu baseline, teachera i distillation."""

from .trainer import BaselineTrainer
from .teacher import TeacherWorkflow
from .distillation import DistillationTrainer

__all__ = ["BaselineTrainer", "TeacherWorkflow", "DistillationTrainer"]
