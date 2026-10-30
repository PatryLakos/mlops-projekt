from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/breast_cancer.csv")
TARGET = "target"


def load_data(path=DATA_PATH):
    # Dane wersjonowane przez DVC (dvc pull); bez pliku - zbior z scikit-learn
    if Path(path).exists():
        dane = pd.read_csv(path)
        return dane.drop(columns=[TARGET]).to_numpy(), dane[TARGET].to_numpy()
    dane = load_breast_cancer()
    return dane.data, dane.target


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
