# Src Pipeline (Reusable)

Ten katalog zawiera uniwersalne klasy do:
- przygotowania danych,
- tokenizacji,
- treningu baseline,
- treningu teachera,
- distillation CE + KL,
- quantization INT8,
- benchmarku CPU.

Przyklad uruchomienia:

```bash
python -m src.cli --config configs/base.yaml prepare-data
python -m src.cli --config configs/base.yaml train-baseline
python -m src.cli --config configs/base.yaml train-teacher
python -m src.cli --config configs/base.yaml distill
python -m src.cli --config configs/base.yaml quantize
python -m src.cli --config configs/base.yaml benchmark
```

Wariant bez nadpisywania artefaktow z notebookow:

```bash
python -m src.cli --config configs/base_cli.yaml prepare-data
python -m src.cli --config configs/base_cli.yaml train-baseline
python -m src.cli --config configs/base_cli.yaml train-teacher
python -m src.cli --config configs/base_cli.yaml distill
python -m src.cli --config configs/base_cli.yaml quantize
python -m src.cli --config configs/base_cli.yaml benchmark
```

Uwaga:
- komentarze w kodzie sa po polsku bez polskich znakow,
- klasy sa projektowane tak, aby latwo podmieniac modele, dane i parametry.
