# Projekt MLOps

**Autor:** Patryk Kaniosz  
**Numer indeksu:** 99564

Klasyfikacja nowotworów (Breast Cancer Wisconsin) – projekt rozwijany przez kolejne laboratoria:
środowisko i DVC (lab 1), strojenie Optuną (lab 2), MLflow (lab 3), jakość kodu i automatyzacja (lab 4),
REST API (lab 5), konteneryzacja i Docker Compose (lab 6).

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
│   ├── config.py             # lab 6 – wspólny odczyt config.yaml (+ MLFLOW_URI z env)
│   ├── api.py                # lab 5 – FastAPI (/health, /predict)
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
├── config.yaml               # lab 6 – konfiguracja API, treningu i strojenia
├── requirements-api.txt      # lab 6 – zależności uruchomieniowe API
├── .dockerignore             # lab 6
├── Dockerfile                # lab 6 – budowa wieloetapowa, użytkownik nieuprzywilejowany
├── docker-compose.yml        # lab 6 – mlflow + api + db (PostgreSQL)
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

---

## Laboratorium 6 – konteneryzacja i Docker Compose

```bash
docker build -t projekt-api:0.2 .          # obraz API (Dockerfile wieloetapowy)
docker volume create mlflow-data           # tylko jeśli wolumen z lab 3 nie istnieje
docker compose up -d --build               # mlflow + api + db
docker compose ps                          # mlflow i db -> healthy
curl http://127.0.0.1:8000/health          # {"status":"ok","model_zaladowany":true}
docker compose exec db psql -U mlflow -d mlflow -c "SELECT version();"
docker compose down                        # usuwa kontenery i sieć, wolumeny zostają
```

- **Konfiguracja** – `config.yaml` jest jedynym źródłem prawdy dla `src/api.py`, `src/train.py`
  i `src/tune.py` (adres MLflow, nazwa modelu, alias, liczba cech, eksperyment, parametry domyślne,
  storage i liczba prób Optuny). Odczyt w `src/config.py` (`yaml.safe_load`), adres serwera można
  nadpisać zmienną `MLFLOW_URI` (w Compose: `http://mlflow:5000`).
- **Użytkownik nieuprzywilejowany** – finalny etap tworzy użytkownika `aplikacja` (UID 1000)
  i przełącza się na niego instrukcją `USER`. Kod i `config.yaml` należą do `root`, więc proces
  aplikacji może je tylko czytać (`docker run --rm projekt-api:0.2 id` → `uid=1000(aplikacja)`).

### Ćwiczenie samodzielne 1 – rozmiary obrazów

| Obraz | Wariant | Rozmiar na dysku (`DISK USAGE`) | Rozmiar skompresowany (`CONTENT SIZE`) |
|---|---|---|---|
| `projekt-api:0.1` | jednoetapowy (`pip install -r ...`) | 1.68 GB | 489 MB |
| `projekt-api:0.2` | wieloetapowy (`--no-cache-dir --prefix=/install`) | 1.26 GB | 286 MB |

Różnica ok. **420 MB (−25%)** na dysku i ok. **200 MB (−41%)** w rejestrze. Źródło: w wariancie
jednoetapowym warstwa `RUN pip install` (1.06 GB wg `docker history`) zawiera oprócz pakietów
także pamięć podręczną pip (`/root/.cache/pip` – pobrane koła, m.in. mlflow, scikit-learn,
numpy, scipy, pandas) i pliki tymczasowe instalacji. Każda warstwa jest niezmienna, więc to,
co raz do niej trafiło, zostaje w obrazie. W budowie wieloetapowej pakiety instalowane są
w etapie `builder` do `/install`, a do finalnego obrazu kopiowany jest tylko ten katalog –
cała reszta etapu budowania (cache pip, metadane, narzędzia) jest porzucana.
Większość pozostałego rozmiaru to same zależności (mlflow ciągnie m.in. pandas, scipy,
pyarrow) oraz obraz bazowy `python:3.12-slim`.

*Pomiar: `docker images projekt-api`, Docker 29 (containerd snapshotter – stąd dwie kolumny).*

### Pytania pomocnicze

1. **Obraz vs kontener.** Obraz to niezmienny, warstwowy szablon (system plików + metadane),
   kontener to uruchomiony na jego podstawie, izolowany proces z cienką warstwą zapisywalną.
   Między środowiskami przenoszony jest **obraz** (przez rejestr).
2. **Zależności przed kodem.** Docker buforuje warstwy i przebudowuje tylko zmienioną warstwę
   i wszystkie po niej. Zależności zmieniają się najrzadziej, kod najczęściej – zmiana kodu nie
   unieważnia kosztownej warstwy `pip install`.
3. **`.dockerignore`.** Wyklucza pliki z kontekstu budowania. Bez niego do silnika trafiałyby
   `.git`, dane, cache DVC, `mlruns/` – wolniejsza budowa, większy obraz (`COPY . .`), ryzyko
   wycieku sekretów/danych i niepotrzebne unieważnianie cache warstw.
4. **Plik vs zmienne środowiskowe.** W pliku – to, co wspólne dla wszystkich uruchomień (nazwa
   modelu, alias, liczba cech, parametry). Nadpisywalne z zewnątrz – to, co zależy od miejsca
   uruchomienia (adres MLflow, hasła, porty). Kryterium: czy wartość zmienia się między
   środowiskami.
5. **`yaml.safe_load`.** Tworzy tylko podstawowe typy (dict, list, str, liczby); `yaml.load`
   z pełnym loaderem potrafi tworzyć dowolne obiekty Pythona, czyli wykonać kod osadzony w pliku.
6. **`0.0.0.0` zamiast `127.0.0.1`.** `127.0.0.1` w kontenerze to interfejs pętli zwrotnej samego
   kontenera; ruch przekierowany przez `-p` przychodzi na jego interfejs sieciowy, więc serwer
   musi nasłuchiwać na wszystkich interfejsach.
7. **Budowa wieloetapowa.** Do finalnego obrazu trafia tylko to, co skopiowano `COPY --from=builder`
   (zainstalowane pakiety) oraz konfiguracja i kod. Redukcja wynika z porzucenia etapu budowania
   (cache pip, narzędzia, kompilatory).
8. **Sieć Compose.** Compose tworzy sieć z wbudowanym DNS – usługi widzą się pod nazwami z pliku
   (`mlflow`, `db`). `host.docker.internal` służył do wyjścia z kontenera na komputer, a teraz
   MLflow działa w tej samej sieci.
9. **`depends_on` + `healthcheck`.** Samo `depends_on` ustala kolejność startu kontenerów, nie
   gotowość usługi. `condition: service_healthy` czeka, aż test `healthcheck` zakończy się
   sukcesem – API pobiera model dopiero z działającego serwera.
10. **Cykl życia wolumenu.** Wolumen istnieje niezależnie od kontenera (przetrwa `stop`, `rm`,
    `compose down`). Dla PostgreSQL to warunek trwałości danych – bez wolumenu usunięcie kontenera
    kasuje bazę.
11. **Brak `ports` w `db`.** Baza jest dostępna dla usług w sieci Compose, ale nie z zewnątrz
    komputera – mniejsza powierzchnia ataku (szczególnie przy haśle zapisanym jawnie w YAML).
