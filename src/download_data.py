"""Pobranie zbiorow danych do katalogu data/ (laboratorium 1).

Uruchomienie z katalogu glownego repozytorium:
    python -m src.download_data
"""

from pathlib import Path

from sklearn.datasets import fetch_california_housing, load_breast_cancer

DATA_DIR = Path("data")


def main():
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)

    # Cwiczenie 5: zbior housing wersjonowany przez DVC
    housing = fetch_california_housing(as_frame=True).frame
    housing.to_csv(DATA_DIR / "housing.csv", index=False)

    # Cwiczenie samodzielne 2: drugi plik danych (uzywany od lab. 4 w src/data.py)
    cancer = load_breast_cancer(as_frame=True).frame
    cancer.to_csv(DATA_DIR / "raw" / "breast_cancer.csv", index=False)

    print("Zapisano data/housing.csv oraz data/raw/breast_cancer.csv")


if __name__ == "__main__":
    main()
