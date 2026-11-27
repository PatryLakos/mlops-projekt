import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import numpy as np
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Sciezka do pliku konfiguracyjnego w katalogu glownym projektu
SCIEZKA_KONFIGURACJI = Path(__file__).resolve().parent.parent / "config.yaml"

with open(SCIEZKA_KONFIGURACJI, encoding="utf-8") as plik:
    KONFIGURACJA = yaml.safe_load(plik)

# Zmienna srodowiskowa ma pierwszenstwo przed wartoscia z pliku
MLFLOW_URI = os.getenv("MLFLOW_URI", KONFIGURACJA["mlflow"]["tracking_uri"])
NAZWA_MODELU = KONFIGURACJA["model"]["nazwa"]
ALIAS_MODELU = KONFIGURACJA["model"]["alias"]
LICZBA_CECH = KONFIGURACJA["model"]["liczba_cech"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ladowanie modelu przy starcie aplikacji i sprzatanie przy zamknieciu."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    app.state.model = mlflow.sklearn.load_model(
        f"models:/{NAZWA_MODELU}@{ALIAS_MODELU}"
    )
    yield
    app.state.model = None


app = FastAPI(
    title="API projektu semestralnego",
    description="Serwowanie modelu klasyfikacyjnego z rejestru MLflow",
    version="1.0.0",
    lifespan=lifespan,
)


class ZadaniePredykcji(BaseModel):
    cechy: list[float] = Field(
        ...,
        min_length=LICZBA_CECH,
        max_length=LICZBA_CECH,
        description="Wektor cech pojedynczej obserwacji",
    )


class OdpowiedzPredykcji(BaseModel):
    klasa: int = Field(description="Przewidziana etykieta klasy")
    prawdopodobienstwo: float = Field(
        ge=0.0,
        le=1.0,
        description="Prawdopodobienstwo przewidzianej klasy",
    )


@app.get("/health")
def health():
    model_zaladowany = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_zaladowany": model_zaladowany}


@app.post("/predict", response_model=OdpowiedzPredykcji)
def predict(zadanie: ZadaniePredykcji):
    obserwacja = np.array(zadanie.cechy).reshape(1, -1)
    klasa = int(app.state.model.predict(obserwacja)[0])
    prawdopodobienstwa = app.state.model.predict_proba(obserwacja)[0]
    return OdpowiedzPredykcji(
        klasa=klasa,
        prawdopodobienstwo=float(prawdopodobienstwa[klasa]),
    )
