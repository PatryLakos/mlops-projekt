"""Wspolny odczyt konfiguracji projektu z pliku config.yaml.

Jedno zrodlo prawdy dla API, treningu i strojenia. Adres serwera MLflow
mozna nadpisac zmienna srodowiskowa MLFLOW_URI (The Twelve-Factor App).
"""

import os
from pathlib import Path

import yaml

# Sciezka do pliku konfiguracyjnego w katalogu glownym projektu
SCIEZKA_KONFIGURACJI = Path(__file__).resolve().parent.parent / "config.yaml"

with open(SCIEZKA_KONFIGURACJI, encoding="utf-8") as plik:
    KONFIGURACJA = yaml.safe_load(plik)

# Zmienna srodowiskowa ma pierwszenstwo przed wartoscia z pliku
MLFLOW_URI = os.getenv("MLFLOW_URI", KONFIGURACJA["mlflow"]["tracking_uri"])
NAZWA_MODELU = KONFIGURACJA["model"]["nazwa"]
