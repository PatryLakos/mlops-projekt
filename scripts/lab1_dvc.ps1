# Laboratorium 1 - DVC. Uruchamiac z katalogu glownego repozytorium,
# w aktywnym srodowisku conda:  conda activate mlops
# Polecenia mozna tez wklejac po kolei recznie.

# --- Cwiczenie 5.1: inicjalizacja DVC ---
dvc init
git commit -m "Inicjalizacja DVC"

# --- Cwiczenie 5.2: pobranie danych (housing.csv + breast_cancer.csv) ---
python -m src.download_data

# --- Cwiczenie 5.3-5.4: housing.csv pod kontrola DVC ---
dvc add data/housing.csv
git add data/housing.csv.dvc data/.gitignore
git commit -m "Dane housing pod kontrola DVC"

# --- Cwiczenie 5.5: lokalny remote (domyslny) ---
dvc remote add -d lokalny C:\dvc-store
git commit .dvc/config -m "Konfiguracja remote DVC"
dvc push

# --- Cwiczenie 5.6: test odtworzenia danych ---
Remove-Item data\housing.csv
dvc pull
dvc status

# --- Cwiczenie samodzielne 2: drugi remote (Dysk Google / dysk sieciowy) ---
# Wariant A - Dysk Google zsynchronizowany jako dysk (np. G:\):
dvc remote add drugi "G:\Moj dysk\dvc-store"
# Wariant B - natywny Google Drive (wymaga: pip install dvc-gdrive i ID folderu):
# dvc remote add drugi gdrive://<ID_FOLDERU>
dvc add data/raw/breast_cancer.csv
git add data/raw/breast_cancer.csv.dvc data/raw/.gitignore .dvc/config
git commit -m "Drugi zbior danych i drugi remote DVC"
dvc push -r drugi

# --- Cwiczenie 5.7: tag i push ---
git push
git tag lab-01-done
git push --tags
