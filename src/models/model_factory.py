from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelFactory:
    @staticmethod
    def load_classifier(
        model_source: str,
        num_labels: int,
        device: torch.device | str = "cpu",
    ) -> AutoModelForSequenceClassification:
        # Wczytujemy model klasyfikacyjny i przenosimy na urzadzenie.
        model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=num_labels)
        model.to(device)
        return model

    @staticmethod
    def load_tokenizer(model_source: str) -> AutoTokenizer:
        # Wczytujemy tokenizer powiazany z modelem.
        return AutoTokenizer.from_pretrained(model_source)
