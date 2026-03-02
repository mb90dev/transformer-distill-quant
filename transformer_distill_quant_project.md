# Transformer Distillation + Quantization (CPU) — projekt pod “AI Engineer” (PyTorch + Transformers)

Ten projekt jest celowo **mały, ale “mocny”**: pokazuje, że rozumiesz trening w PyTorch (bez HF Trainer), distillation (logity/softmax/temperatura/KL) i produkcyjne kompromisy (latency/pamięć/throughput) na **CPU**.

> **Cel**: zbudować i porównać 3 warianty modelu klasyfikacji tekstu:
> 1) **Teacher** (większy / lepsza jakość)  
> 2) **Student** (mniejszy / szybszy) trenowany **z distillation**  
> 3) **Student INT8** (quantized) + benchmark wydajności na CPU

---

## TL;DR (co będzie w repo)

- **Własna pętla treningowa** w PyTorch: forward → loss → backward → grad clipping → optimizer.step → zero_grad  
- **Distillation loss**: CE + KL(student || teacher) na logitach z temperaturą  
- **Quantization**: dynamic INT8 (PyTorch) i porównanie z FP32  
- **Benchmark**: latency p50/p95, throughput, batch=1 vs batch>1, warmup  
- **Raport** (README / report.md) z tabelami i wnioskami jak w “engineering study”

---

## Dlaczego to ma sens do CV?

Bo w praktyce firmy chcą ludzi, którzy nie tylko “trenują model”, ale też potrafią go **uczynić tańszym i szybszym** oraz potrafią to **zmierzyć**.

---

## Zbiór danych (CPU-friendly)

### Dataset: **Financial PhraseBank** (Hugging Face `datasets`)
- Zadanie: klasyfikacja sentymentu finansowego (zwykle 3 klasy: negative/neutral/positive)
- Mały i idealny do CPU

**Uwaga**: jeśli chcesz jeszcze prostszy dataset do sanity-checku, możesz dodać `SST-2`, ale główny benchmark rób na PhraseBank.

---

## Modele (CPU-friendly)

### Teacher (większy, ale nadal sensowny na CPU)
- `distilroberta-base` **albo** `bert-base-uncased`  
Rekomendacja CPU: **distilroberta-base** (zwykle stabilny i nie tak ciężki jak DeBERTa)

### Student (wyraźnie mniejszy)
- `prajjwal1/bert-tiny` (super szybki)  
**albo**
- `microsoft/MiniLM-L12-H384-uncased` (trochę większy, nadal lekki)

**Rekomendacja**: zacznij od `bert-tiny` (żeby szybko zobaczyć efekt), potem ewentualnie zrób wersję z MiniLM jako “bonus”.

---

## Repo — przykładowa struktura

```text
transformer-distill-quant/
│
├─ README.md
├─ report.md
├─ requirements.txt
├─ configs/
│   ├─ base.yaml
│   ├─ teacher.yaml
│   ├─ student.yaml
│   └─ bench.yaml
│
├─ src/
│   ├─ data/
│   │   ├─ load_dataset.py
│   │   └─ tokenize.py
│   │
│   ├─ models/
│   │   ├─ build_model.py
│   │   └─ quantize.py
│   │
│   ├─ training/
│   │   ├─ losses.py            # CE, KL distillation, temperature softmax
│   │   ├─ train_teacher.py     # opcjonalnie
│   │   ├─ distill_student.py   # główna część
│   │   └─ utils.py             # seed, device, grad norms
│   │
│   ├─ eval/
│   │   ├─ metrics.py           # accuracy, f1
│   │   └─ evaluate.py
│   │
│   ├─ bench/
│   │   ├─ latency.py           # warmup, p50/p95, throughput
│   │   └─ memory.py            # size on disk, (opcjonalnie) RSS
│   │
│   └─ cli.py                   # prosty CLI: train/distill/quantize/bench
│
└─ artifacts/
   ├─ teacher/
   ├─ student_fp32/
   └─ student_int8/
```

---

# Krok po kroku (plan wykonania)

Poniżej masz plan w kolejności, w jakiej warto to robić.  
Każdy etap kończy się “outputem”, który możesz commitować.

---

## Etap 0 — Setup (CPU)

1) Utwórz venv i zainstaluj zależności.
2) Upewnij się, że PyTorch działa na CPU.
3) Dodaj `requirements.txt` (minimum):
- torch
- transformers
- datasets
- numpy
- scikit-learn
- tqdm
- pyyaml

**Output**: działający `python -c "import torch; print(torch.__version__)"`

---

## Etap 1 — Dane + tokenizacja

### 1.1 Pobranie datasetu
- Użyj `datasets.load_dataset(...)`
- Zrób split: train/val/test (jeśli dataset nie ma wprost)

### 1.2 Tokenizacja
- Tokenizer zgodny z teacher/student (dla uproszczenia możesz użyć tokenizer teachera i student też go “łyka” — ale najlepiej trzymać pary):
  - teacher tokenizer dla teachera
  - student tokenizer dla studenta

Ustaw:
- `max_length = 128`
- `padding = "max_length"`
- `truncation = True`

**Output**: `tokenized_train.pt` (opcjonalnie) + szybki sanity-check batcha.

---

## Etap 2 — Baseline: student fine-tuning (bez distillation)

Zanim distillation, zrób baseline:
- student trenuje tylko na klasycznych labelach (CrossEntropy)

### Własna pętla treningowa (PyTorch)
W pętli muszą być jawnie:
- `model(**batch)` / forward
- `loss = CE(logits, labels)`
- `loss.backward()`
- `clip_grad_norm_`
- `optimizer.step()`
- `optimizer.zero_grad(set_to_none=True)`

**Output**:
- zapisany model: `artifacts/student_fp32_baseline/`
- metryki: Accuracy/F1 na walidacji/test

---

## Etap 3 — Teacher (dwie opcje)

### Opcja A (polecana na CPU): teacher “bez treningu”
- Bierz pretrained teacher i używasz go tylko do generowania logitów.
- Plusem jest zero dodatkowego treningu.

### Opcja B (opcjonalnie): krótki fine-tuning teachera
- 1 epoka, mały LR, max_length=128, batch mały.
- Zwykle poprawia jakość i robi distillation “ładniejsze”.

**Output**:
- `artifacts/teacher/` (zapisane wagi)
- log: metryki teachera (jeśli trenowałeś)

---

## Etap 4 — Distillation (serce projektu)

### Co robi student?
Student uczy się z dwóch źródeł:
1) **prawdziwe etykiety** (CE)
2) **“miękkie etykiety” teachera** (KL)

### Kluczowe pojęcia (które masz umieć wytłumaczyć)
- **logits**: surowe wyjście modelu przed softmax
- **temperature (T)**: “zmiękcza” softmax
- **KL divergence**: “jak bardzo rozkład studenta różni się od teachera”

### Distillation loss (klasyk)
- `p_teacher = softmax(teacher_logits / T)`
- `p_student = log_softmax(student_logits / T)`
- `loss_kd = KL(p_student, p_teacher) * (T^2)`
- `loss = alpha * CE(student_logits, labels) + (1-alpha) * loss_kd`

Typowe wartości:
- `T = 2..4`
- `alpha = 0.2..0.7` (większe alpha → bardziej “pod etykiety”, mniejsze → bardziej “naśladowanie teachera”)

**Output**:
- `artifacts/student_fp32_distilled/`
- metryki student distilled vs baseline student

---

## Etap 5 — Quantization (INT8) na CPU

Użyj **dynamic quantization** (najłatwiejsze na CPU):
- quantyzujesz głównie warstwy `Linear`

W PyTorch zwykle wygląda to jak:
- `torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`

**Output**:
- `artifacts/student_int8/`
- metryki jakości po quantization (często minimalny spadek)
- porównanie rozmiaru na dysku

---

## Etap 6 — Benchmark (latency, p50/p95, throughput, batch)

To jest element, który robi projekt “AI engineer”.

### Zasady dobrego benchmarku
- **Warmup**: np. 50–200 iteracji przed pomiarem
- Pomiar osobno dla:
  - batch=1 (online inference)
  - batch=8/16 (batch inference)
- Mierz:
  - p50, p95
  - throughput (req/s)
- Ustal stały input (ten sam max_length)

### Co porównujesz
- Teacher FP32 vs Student FP32 vs Student INT8
- Batch=1 i Batch>1

**Output**:
- `bench_results.json` + tabelka w `report.md`

---

## Etap 7 — Raport i “CV story”

W `report.md` / `README.md` daj:

1) **Cel** i ograniczenia (CPU-only)
2) **Metody** (distillation + quantization)
3) **Tabela wyników**: jakość vs latency vs size
4) **Wnioski**:
   - gdzie student wygrywa
   - ile kosztuje quantization w jakości
   - kiedy batch inference ma sens
5) “Lessons learned” (krótko)

---

# Co realnie nauczysz się dzięki temu projektowi (mapowanie na Twoje wymagania)

## ✅ 1) Jak działa trening w PyTorch
W projekcie jawnie robisz:
- forward pass
- loss (CE oraz distillation KL)
- backward
- gradient flow (możesz logować normy gradientów)
- optimizer.step()
- zero_grad(set_to_none=True)
- gradient clipping

## ✅ 2) Logity i softmax
W distillation operujesz na:
- logits
- softmax z temperaturą
- log_softmax
- KL divergence

To jest “wyższy level” niż zwykłe CE.

## ✅ 3) Pamięć i typy wag (quantization)
Zobaczysz praktycznie:
- FP32 vs INT8
- różnica w rozmiarze modelu
- wpływ na latency

## ✅ 4) Sensowny benchmark
Nauczysz się:
- batch=1 vs batch>1
- warmup
- p50/p95
- throughput

---

# Czego ten projekt NIE uczy (i czemu to OK)

❌ Nie piszesz transformera od zera  
❌ Nie implementujesz self-attention ręcznie  
❌ Nie wchodzisz głęboko w matematykę architektury

Ale **AI Engineer** zwykle nie musi pisać attention od zera — to bardziej research / infra.

---

# “Poziomy PyTorch” — gdzie jesteś po tym projekcie?

🟢 Poziom 1 — Użytkownik  
- Trainer, gotowe pipeline’y, copy-paste

🟡 Poziom 2 — Inżynier (**to osiągasz tym projektem**)  
- custom training loop  
- kontrola optymalizacji i lossów (CE + KL)  
- freezing layers (opcjonalny eksperyment)  
- quantization  
- benchmark wydajności

🔴 Poziom 3 — Architekt  
- attention od zera, custom kernels, research

Pod AI Engineer role w 2026 zwykle **wystarczy poziom 2**.

---

# Opcjonalne “bonusy” (jeśli będziesz chciał)

### Bonus 1 — Layer freezing
Dodaj eksperyment:
- tylko head
- ostatnie 2 warstwy
- cały model  
Porównaj jakość i czas.

### Bonus 2 — Gradient norms
Loguj normy gradientów w czasie (np. co N batchy).

### Bonus 3 — ONNX / TorchScript
Eksportuj student model i porównaj latency.

---

# Minimalny scope (żeby nie przedobrzyć)

Jeśli chcesz to dowieźć szybko:
- Student baseline (CE)
- Distilled student (CE + KL)
- Quantization studenta
- Benchmark: batch=1 i batch=8, p50/p95 + throughput
- Jedna tabelka i konkretne wnioski

---

## Proponowane configi (startowe)

- `max_length=128`
- `batch_size=8`
- `epochs=1..3`
- `lr=2e-5` (student), `lr=1e-5` (teacher jeśli trenujesz)
- `T=3`
- `alpha=0.5`
- `grad_clip=1.0`

---

## Jak to brzmi w CV (przykładowy bullet)

- Implemented custom PyTorch training loop for transformer distillation (CE + KL with temperature) and applied INT8 dynamic quantization to optimize CPU inference; measured latency p50/p95 and throughput, achieving significant speedup with minimal F1 drop.

---

## Następny krok
1) Skopiuj strukturę repo
2) Zrób Etap 1 (dane + tokenizacja)
3) Zrób Etap 2 (student baseline, custom loop)
4) Dopiero potem distillation i quantization

Powodzenia — to jest mały projekt, ale świetnie “domyka” Twój profil AI Engineer.
