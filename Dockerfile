# Obraz bazowy z interpreterem Python
FROM python:3.12-slim

# Katalog roboczy wewnatrz obrazu
WORKDIR /app

# Najpierw zaleznosci, aby wykorzystac pamiec podreczna warstw
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Konfiguracja i kod aplikacji
COPY config.yaml .
COPY src/ ./src/

# Deklaracja portu, na ktorym nasluchuje aplikacja
EXPOSE 8000

# Polecenie uruchamiane przy starcie kontenera
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]