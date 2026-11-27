# Etap 1: budowanie, instalacja zaleznosci do wydzielonego katalogu
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# Etap 2: finalny obraz uruchomieniowy
FROM python:3.12-slim

WORKDIR /app

# Uzytkownik nieuprzywilejowany (cwiczenie samodzielne 1), bez powloki logowania
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin aplikacja

# Przeniesienie gotowych pakietow z etapu budowania
COPY --from=builder /install /usr/local

# Konfiguracja i kod aplikacji (wlasnosc root, uzytkownik ma tylko odczyt)
COPY config.yaml .
COPY src/ ./src/

# Od tego miejsca wszystkie polecenia, w tym CMD, dzialaja jako "aplikacja"
USER aplikacja

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
