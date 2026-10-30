# Projekt MLOps

**Autor:** Patryk Kaniosz  
**Numer indeksu:** 99564

Klasyfikacja nowotworów (Breast Cancer Wisconsin) – projekt rozwijany przez kolejne laboratoria:
środowisko i DVC (lab 1), strojenie Optuną (lab 2), MLflow (lab 3), jakość kodu i automatyzacja (lab 4).

## Struktura

```
mlops-projekt/
├── .dvc/                     # konfiguracja DVC (po `dvc init`)
├── data/
│   ├── housing.csv(.dvc)     # lab 1 – dane pod kontrolą DVC
│   └── raw/breast_cancer.csv(.dvc)  # lab 1 (ćw. sam. 2), używany przez src/data.py
├── notebooks/
│   ├── lab2_optuna.ipynb     # lab 2 – Optuna
│   └── lab3_mlflow.ipynb     # lab 3 – brudnopis
├── scripts/
│   ├── lab1_dvc.ps1          # lab 1 – polecenia DVC/Git
│   └── lab3_mlflow_server.ps1  # lab 3 – serwer MLflow w Dockerze
├── src/
│   ├── __init__.py
│   ├── data.py               # lab 4 – wczytywanie i podział danych
│   ├── train.py              # lab 4 – trening finalny + rejestracja w MLflow
│   ├── tune.py               # lab 4 – strojenie Optuna + MLflowCallback
│   ├── evaluate.py           # lab 4 – ewaluacja
│   ├── download_data.py      # lab 1 – pobranie danych
│   └── train_tracked.py, tune_tracked.py, register_model*.py,
│       set_alias.py, predict_from_registry.py,
│       compare_versions.py, tune_nested.py   # lab 3
├── tests/
│   ├── __init__.py
│   └── test_model.py         # lab 4 – 5 testów (3 z instrukcji + 2 własne)
├── Makefile
├── pyproject.toml            # konfiguracja ruff i pytest
├── .pre-commit-config.yaml
├── requirements.txt
└── environment.yml
```

## Odtworzenie środowiska

```bash
conda env create -f environment.yml
conda activate mlops
pip install -r requirements.txt
dvc pull
```

---

## Laboratorium 1 – środowisko, Git, DVC

Polecenia DVC: [scripts/lab1_dvc.ps1](scripts/lab1_dvc.ps1).

### Ćwiczenie samodzielne 1 – porównanie narzędzi

Czasy instalacji `numpy scikit-learn` należy wpisać na podstawie własnych pomiarów
(`Measure-Command { ... }`).

| Narzędzie | Czas instalacji | Plik z definicją zależności | Odtworzenie jedną komendą | Kiedy stosować |
|---|---|---|---|---|
| venv + pip | _… s_ | `requirements.txt` | `pip install -r requirements.txt` | Proste projekty czysto Pythonowe, brak dodatkowych narzędzi. |
| uv | _… s_ | `requirements.txt` / `pyproject.toml` (+ `uv.lock`) | `uv pip install -r requirements.txt` / `uv sync` | Domyślny wybór dla projektów Pythonowych – wielokrotnie szybszy od pip, lockfile. |
| conda | _… s_ | `environment.yml` | `conda env create -f environment.yml` | Gdy potrzebne są zależności spoza PyPI: CUDA, MKL, kompilatory, konkretna wersja interpretera. |
| **DVC** | – | `*.dvc` + `.dvc/config` | `dvc pull` | Wersjonuje **dane i modele** (duże pliki). Git przechowuje tylko lekki wskaźnik z hashem md5, a plik trafia do zdalnego magazynu (remote). Git wersjonuje kod. |

### Ćwiczenie samodzielne 2 – drugi remote

Drugi remote `drugi` (Dysk Google / dysk sieciowy) oraz drugi plik `data/raw/breast_cancer.csv`
wypchnięty poleceniem `dvc push -r drugi` – patrz koniec [scripts/lab1_dvc.ps1](scripts/lab1_dvc.ps1).

### Ćwiczenie samodzielne 3 – odtworzenie repozytorium kolegi/koleżanki

```bash
git clone https://github.com/<LOGIN>/mlops-projekt.git kolega-projekt
cd kolega-projekt
conda env create -f environment.yml
conda activate mlops
dvc pull
```

Napotkane problemy: _(uzupełnić, np. brak dostępu do remote wskazującego lokalny katalog
`C:\dvc-store` na innym komputerze – `dvc pull` nie zadziała bez wspólnego magazynu; pole
`prefix:` w `environment.yml` ze ścieżką autora; pakiety przypięte do wersji niedostępnych na innym OS)._

### Pytania pomocnicze

1. **Dlaczego `.venv` nie trafia do Gita?** Zawiera binaria zależne od systemu i ścieżek, waży setki MB
   i nie jest przenośne. Odtwarzalność zapewnia plik z listą zależności (`requirements.txt`,
   `environment.yml`), z którego środowisko odtwarza się jedną komendą.
2. **`requirements.txt` vs `environment.yml`.** Pierwszy to lista pakietów PyPI (pip, uv). Drugi
   (conda) opisuje całe środowisko: wersję Pythona, kanały, pakiety niepythonowe (MKL, CUDA) i może
   zawierać sekcję pip.
3. **Kiedy conda?** Gdy potrzebne są biblioteki systemowe/natywne, np. trening sieci w PyTorch
   na GPU z konkretną wersją CUDA i cuDNN – conda instaluje je razem z Pythonem bez ręcznej
   konfiguracji sterowników bibliotek.
4. **Plik `.dvc`.** Zawiera hash md5 zawartości, rozmiar, liczbę plików i ścieżkę. To tylko wskaźnik
   na dane w magazynie, dlatego jego rozmiar (kilka linii YAML) nie zależy od rozmiaru danych.
5. **Zespół na 3 systemach.** Kod i definicje środowiska w Gicie (`environment.yml` bez `prefix`,
   przypięte wersje / lockfile `uv.lock`), dane w DVC ze wspólnym remote (S3, GDrive), konkretne
   wersje oznaczone tagami Git (kod + `.dvc` = dane). Najpewniej: obraz Docker (lab 6) i CI (lab 8).

---

## Laboratorium 2 – Optuna

Notebook: [notebooks/lab2_optuna.ipynb](notebooks/lab2_optuna.ipynb) – wszystkie ćwiczenia,
ćwiczenia samodzielne (rozszerzona przestrzeń, tabela porównawcza z komentarzem, interpretacja
wykresów), optymalizacja równoległa z SQLite, porównanie samplerów TPE/Random i odpowiedzi na pytania.

| Model | Dokładność CV | Dokładność testowa |
|---|---|---|
| Bazowy | 0.9538 | 0.9561 |
| Strojony (Optuna, 50 prób) | 0.9604 | 0.9474 |

---

## Laboratorium 3 – MLflow

Serwer: [scripts/lab3_mlflow_server.ps1](scripts/lab3_mlflow_server.ps1) (tam też kolejność skryptów).

- `src/train_tracked.py` – ręczne logowanie parametrów, metryk i macierzy pomyłek,
- `src/tune_tracked.py` – Optuna + `MLflowCallback`,
- `src/register_model.py`, `src/set_alias.py`, `src/predict_from_registry.py` – Model Registry, alias `champion`,
- `src/register_model_v2.py`, `src/compare_versions.py` – ćw. samodzielne 1 (druga wersja – GradientBoosting, przepięcie aliasu wg `f1_macro`),
- `src/tune_nested.py` – ćw. samodzielne 2 (badanie jako run nadrzędny, próby jako nested runs; run nadrzędny loguje `best_cv_accuracy` i `best_*` parametry).

**Korzyść z nested runs:** wszystkie próby jednego badania są zgrupowane pod jednym runem
nadrzędnym, więc w jednym eksperymencie można prowadzić wiele badań (np. różne przestrzenie,
samplery, dni) bez mieszania kilkuset prób na płaskiej liście. Run nadrzędny jest podsumowaniem
badania (najlepszy wynik i parametry), można go zwinąć/rozwinąć, porównywać badania między sobą
i jednym kliknięciem przejść do prób, które je tworzą.

---

## Laboratorium 4 – jakość kodu, testy, Makefile

```bash
make lint      # ruff check src/ tests/
make format    # ruff format + ruff check --fix
make test      # pytest tests/ -v
make tune      # dvc pull + python -m src.tune  (strojenie, zapis do optuna.db)
make train     # dvc pull + python -m src.train (rejestracja modelu w MLflow)
make all       # lint -> test -> train
make clean     # usuwa __pycache__, .pytest_cache, .ruff_cache
```

Na Windows `make` można zainstalować np. przez `conda install -c conda-forge make`
albo `winget install GnuWin32.Make`.

### Ćwiczenie samodzielne 1

- `test_split_data` – liczności podzbiorów i zgodność rozmiaru testu z `test_size` (dla 0.2 i 0.3),
- `test_stratyfikacja_i_odtwarzalnosc` – zachowanie proporcji klas po podziale stratyfikowanym
  i identyczne predykcje przy ustalonym `random_state`,
- cel `clean` w Makefile (dopisany do `.PHONY`).

### Ćwiczenie samodzielne 2 – pre-commit

```bash
pip install pre-commit
pre-commit install
```

Konfiguracja: [.pre-commit-config.yaml](.pre-commit-config.yaml) (hooki `ruff-check` i `ruff-format`
z `astral-sh/ruff-pre-commit`). Demonstracja – dopisz `import os` do `src/data.py` i spróbuj
zatwierdzić zmianę:

```text
(wklej tu wynik `git commit` odrzuconego przez hook)
```

### Pytania pomocnicze

1. **Analiza statyczna vs testy.** Linter bada kod bez uruchamiania (styl, nieużywane importy,
   niezdefiniowane nazwy). Testy uruchamiają kod i sprawdzają, czy wynik jest poprawny. Linter nie
   wykryje błędnej logiki, a testy nie wykryją niespójnego stylu czy martwego kodu – potrzebne są oba.
2. **`tune.py` i `train.py` osobno.** Strojenie jest kosztowne i okazjonalne (wynik: najlepsze
   parametry), trening finalny jest szybki i powtarzalny (wynik: model w rejestrze). Połączenie
   powodowałoby uruchamianie dziesiątek prób przy każdym treningu, także w CI.
3. **Dwie role MLflow.** W `tune.py` śledzi próby strojenia (każda próba = run). W `train.py`
   rejestruje finalny model – do Model Registry trafia tylko model wytrenowany na najlepszych
   parametrach, nie modele z prób.
4. **Dane poza `objective`.** Wczytanie i podział wykonywane są raz, a nie w każdej próbie (oszczędność
   czasu) i wszystkie próby oceniane są na identycznym podziale, więc wyniki są porównywalne.
5. **`.PHONY`.** Oznacza cele, które nie są plikami. Bez niej, gdyby w katalogu istniał plik/katalog
   o nazwie celu (np. `test`), make uznałby cel za aktualny i nie wykonałby receptury.
6. **`tune` poza `all`.** `all` służy do szybkiej weryfikacji (także w CI). Strojenie trwa długo,
   jest niedeterministyczne i zmieniałoby parametry modelu przy każdym uruchomieniu – weryfikacja
   byłaby wolna i niepowtarzalna.
