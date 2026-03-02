# Kolejnosc notebookow 01-08

- `01_setup.ipynb` - przygotowanie srodowiska i sanity check CPU
- `02_data_tokenization.ipynb` - pobranie danych, split, tokenizacja
- `03_student_baseline.ipynb` - baseline student (CE, custom training loop)
- `04_teacher_logits.ipynb` - teacher i zapis logits do distillation
- `05_distillation.ipynb` - trening studenta z CE + KL (temperature)
- `06_quantization_int8.ipynb` - dynamic quantization INT8
- `07_benchmark_cpu.ipynb` - benchmark p50/p95/throughput
- `08_report_cv.ipynb` - podsumowanie wynikow i bullet do CV
