from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from .bench.cpu_benchmark import CPUBenchmarkRunner
from .config import ProjectConfig
from .data.dataset_builder import TextClassificationDatasetBuilder
from .data.tokenization import TokenizationPipeline
from .models.model_factory import ModelFactory
from .models.quantization import DynamicQuantizer
from .training.distillation import DistillationTrainer
from .training.teacher import TeacherWorkflow
from .training.trainer import BaselineTrainer
from .training.utils import set_seed


def _set_torch_format(ds) -> None:
    # Wymuszamy format tensorowy po load_from_disk, aby DataLoader dawal torch.Tensor.
    cols = ["input_ids", "attention_mask", "label", "idx"]
    ds.set_format(type="torch", columns=cols)


def cmd_prepare_data(cfg: ProjectConfig) -> None:
    # Budujemy i zapisujemy dane + splity + tokenizacje.
    Path(cfg.paths.artifacts_dir).mkdir(parents=True, exist_ok=True)
    builder = TextClassificationDatasetBuilder(
        label_map=cfg.data.label_map,
        split_seed=cfg.data.split_seed,
        test_size=cfg.data.test_size,
    )
    raw_df = builder.load_dataframe(
        cfg.data.source,
        encoding=cfg.data.encoding,
        separator=cfg.data.separator,
    )
    norm_df = builder.normalize_columns(raw_df, cfg.data.text_column, cfg.data.label_column)
    mapped_df = builder.map_labels(norm_df)

    raw_csv = Path(cfg.paths.data_dir) / "all-data_raw.csv"
    clean_csv = Path(cfg.paths.data_dir) / "all-data_clean.csv"
    builder.save_csv(mapped_df, raw_csv, clean_csv)

    splits = builder.build_splits(mapped_df)
    (Path(cfg.paths.artifacts_dir) / "splits_summary.json").write_text(
        json.dumps({k: len(v) for k, v in splits.items()}, indent=2),
        encoding="utf-8",
    )

    token_pipeline = TokenizationPipeline(
        teacher_model_name=cfg.models.teacher_model_name,
        student_model_name=cfg.models.student_model_name,
        max_length=cfg.data.max_length,
    )
    tokenized_teacher, tokenized_student = token_pipeline.build(splits)
    token_pipeline.save(
        tokenized_teacher,
        tokenized_student,
        cfg.paths.tokenized_teacher_dir,
        cfg.paths.tokenized_student_dir,
    )
    print("Dane i tokenizacja gotowe.")


def cmd_train_baseline(cfg: ProjectConfig) -> None:
    # Trenujemy baseline studenta CE.
    set_seed(cfg.training.seed)
    device = torch.device("cpu")
    ds = load_from_disk(cfg.paths.tokenized_student_dir)
    _set_torch_format(ds)

    train_loader = DataLoader(ds["train"], batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(ds["validation"], batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(ds["test"], batch_size=cfg.training.batch_size, shuffle=False)

    model = ModelFactory.load_classifier(cfg.models.student_model_name, cfg.models.num_labels, device)
    trainer = BaselineTrainer(
        model=model,
        tokenizer_source=cfg.models.student_model_name,
        device=device,
        lr=cfg.training.student_lr,
        grad_clip=cfg.training.grad_clip,
    )
    metrics = trainer.fit(train_loader, val_loader, test_loader, epochs=cfg.training.baseline_epochs)
    trainer.save(Path(cfg.paths.artifacts_dir) / "student_fp32_baseline")
    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.outputs_dir, "metrics_student_baseline.json").write_text(
        json.dumps({"validation": metrics["validation"], "test": metrics["test"]}, indent=2),
        encoding="utf-8",
    )
    print("Baseline gotowy.")


def cmd_quantize(cfg: ProjectConfig) -> None:
    # Quantization modelu studenta.
    distilled = Path(cfg.paths.artifacts_dir) / "student_fp32_distilled"
    baseline = Path(cfg.paths.artifacts_dir) / "student_fp32_baseline"
    source = distilled if distilled.joinpath("config.json").exists() else baseline

    model = ModelFactory.load_classifier(str(source), cfg.models.num_labels, "cpu")
    quantizer = DynamicQuantizer()
    model_int8 = quantizer.quantize_linear(model)
    quantizer.save(model_int8, Path(cfg.paths.artifacts_dir) / "student_int8")
    print("Quantization zakonczony.")


def cmd_train_teacher(cfg: ProjectConfig) -> None:
    # Trenujemy teachera na danych tokenized teachera i zapisujemy model + metryki.
    set_seed(cfg.training.seed)
    device = torch.device("cpu")
    ds = load_from_disk(cfg.paths.tokenized_teacher_dir)
    _set_torch_format(ds)

    train_loader = DataLoader(ds["train"], batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(ds["validation"], batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(ds["test"], batch_size=cfg.training.batch_size, shuffle=False)

    workflow = TeacherWorkflow(
        teacher_model_name=cfg.models.teacher_model_name,
        num_labels=cfg.models.num_labels,
        device=device,
        lr=cfg.training.teacher_lr,
        epochs=cfg.training.teacher_epochs,
        grad_clip=cfg.training.grad_clip,
    )
    workflow.load()
    # Ustawiamy wysoki prog, aby w praktyce wymusic fine-tuning teachera w CLI.
    metrics = workflow.finetune_if_needed(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_test_f1=1.0,
        enable_finetune=True,
    )

    teacher_dir = Path(cfg.paths.artifacts_dir) / "teacher_finetuned"
    workflow.save_model(teacher_dir, safe_serialization=False)

    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    workflow.save_metrics(metrics, Path(cfg.paths.outputs_dir) / "metrics_teacher.json")
    print("Teacher gotowy.")


def cmd_distill(cfg: ProjectConfig) -> None:
    # Trenujemy studenta w trybie KD (CE + KL) i zapisujemy najlepszy/finalny checkpoint.
    set_seed(cfg.training.seed)
    device = torch.device("cpu")

    student_ds = load_from_disk(cfg.paths.tokenized_student_dir)
    teacher_ds = load_from_disk(cfg.paths.tokenized_teacher_dir)
    _set_torch_format(student_ds)
    _set_torch_format(teacher_ds)

    train_loader = DataLoader(student_ds["train"], batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(student_ds["validation"], batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(student_ds["test"], batch_size=cfg.training.batch_size, shuffle=False)

    teacher_finetuned = Path(cfg.paths.artifacts_dir) / "teacher_finetuned"
    teacher_saved = Path(cfg.paths.artifacts_dir) / "teacher"
    if teacher_finetuned.joinpath("config.json").exists():
        teacher_source = teacher_finetuned
    elif teacher_saved.joinpath("config.json").exists():
        teacher_source = teacher_saved
    else:
        teacher_source = cfg.models.teacher_model_name

    logits_dir = Path(cfg.paths.artifacts_dir) / "teacher_logits"
    if cfg.distill.alpha < 1.0:
        teacher_workflow = TeacherWorkflow(
            teacher_model_name=cfg.models.teacher_model_name,
            num_labels=cfg.models.num_labels,
            device=device,
            lr=cfg.training.teacher_lr,
            epochs=cfg.training.teacher_epochs,
            grad_clip=cfg.training.grad_clip,
        )
        teacher_workflow.load(model_path=teacher_source if isinstance(teacher_source, Path) else None)
        teacher_workflow.save_logits_for_splits(
            tokenized_teacher=teacher_ds,
            output_dir=logits_dir,
            batch_size=cfg.training.batch_size,
        )
        teacher_map = DistillationTrainer.build_teacher_map(logits_dir / "teacher_logits_train.npz")
    else:
        teacher_map = None

    trainer = DistillationTrainer(
        student_model_name=cfg.models.student_model_name,
        num_labels=cfg.models.num_labels,
        device=device,
        lr=cfg.training.student_lr,
        grad_clip=cfg.training.grad_clip,
        temperature=cfg.distill.temperature,
        alpha=cfg.distill.alpha,
        use_warm_start=cfg.distill.use_warm_start,
    )

    warm_start = Path(cfg.paths.artifacts_dir) / "student_fp32_baseline"
    trainer.load_student(warm_start_path=warm_start)

    best_ckpt = Path(cfg.paths.artifacts_dir) / "student_fp32_distilled_best"
    metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        teacher_map=teacher_map,
        epochs=cfg.training.distill_epochs,
        best_ckpt_dir=best_ckpt,
    )

    final_dir = Path(cfg.paths.artifacts_dir) / "student_fp32_distilled"
    trainer.save(final_dir)

    Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)
    DistillationTrainer.save_metrics(
        metrics=metrics,
        metrics_path=Path(cfg.paths.outputs_dir) / "metrics_student_distilled.json",
        history_path=Path(cfg.paths.outputs_dir) / "history_student_distilled.json",
    )
    print("Distillation zakonczona.")


def cmd_benchmark(cfg: ProjectConfig) -> None:
    # Benchmark CPU modeli.
    runner = CPUBenchmarkRunner(
        warmup_steps=cfg.benchmark.warmup_steps,
        measure_steps=cfg.benchmark.measure_steps,
        batches=tuple(cfg.benchmark.batches),
        device=torch.device("cpu"),
    )

    teacher_dir = Path(cfg.paths.artifacts_dir) / "teacher_finetuned"
    if teacher_dir.joinpath("config.json").exists():
        teacher_source = str(teacher_dir)
    else:
        teacher_dir = Path(cfg.paths.artifacts_dir) / "teacher"
        if teacher_dir.joinpath("config.json").exists():
            teacher_source = str(teacher_dir)
        else:
            teacher_source = cfg.models.teacher_model_name

    student_fp32_dir = Path(cfg.paths.artifacts_dir) / "student_fp32_distilled"
    if not student_fp32_dir.joinpath("config.json").exists():
        student_fp32_dir = Path(cfg.paths.artifacts_dir) / "student_fp32_baseline"

    teacher = ModelFactory.load_classifier(teacher_source, cfg.models.num_labels, "cpu")
    student_fp32 = ModelFactory.load_classifier(str(student_fp32_dir), cfg.models.num_labels, "cpu")

    quantizer = DynamicQuantizer()
    int8_state = Path(cfg.paths.artifacts_dir) / "student_int8" / "model_int8_state_dict.pt"
    if not int8_state.exists():
        model_int8 = quantizer.quantize_linear(student_fp32)
        quantizer.save(model_int8, Path(cfg.paths.artifacts_dir) / "student_int8")
    student_int8 = quantizer.load_from_state_dict(student_fp32, int8_state)

    make_batch = runner.build_sample_batch_fn(cfg.paths.tokenized_student_dir)
    results = runner.run_suite(
        {
            "teacher_fp32": teacher,
            "student_fp32": student_fp32,
            "student_int8": student_int8,
        },
        make_batch,
    )
    runner.save_results(results, Path(cfg.paths.outputs_dir) / "bench_results.json")
    print(json.dumps(results, indent=2))


def main() -> None:
    # Proste CLI do uruchamiania glownego pipeline'u bez notebookow.
    parser = argparse.ArgumentParser(description="Transformer distill + quant pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Sciezka do pliku YAML z konfiguracja")
    parser.add_argument(
        "command",
        choices=[
            "prepare-data",
            "train-baseline",
            "train-teacher",
            "distill",
            "quantize",
            "benchmark",
        ],
        help="Krok pipeline do wykonania",
    )
    args = parser.parse_args()

    cfg = ProjectConfig.from_yaml(args.config)

    if args.command == "prepare-data":
        cmd_prepare_data(cfg)
    elif args.command == "train-baseline":
        cmd_train_baseline(cfg)
    elif args.command == "train-teacher":
        cmd_train_teacher(cfg)
    elif args.command == "distill":
        cmd_distill(cfg)
    elif args.command == "quantize":
        cmd_quantize(cfg)
    elif args.command == "benchmark":
        cmd_benchmark(cfg)


if __name__ == "__main__":
    main()
