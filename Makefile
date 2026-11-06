PYTHON = python
KATALOGI = src/ tests/

.PHONY: install data lint format test tune train all clean

install:
	pip install -r requirements.txt

data:
	dvc pull

serve:
	$(PYTHON)-m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

lint:
	ruff check $(KATALOGI)

format:
	ruff format $(KATALOGI)
	ruff check --fix $(KATALOGI)

test:
	pytest tests/ -v

tune: data
	$(PYTHON) -m src.tune

train: data
	$(PYTHON) -m src.train

all: lint test train

# Usuwanie plikow tymczasowych (dziala na Windows i Linux, bez rm -rf)
clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for n in ('__pycache__', '.pytest_cache', '.ruff_cache') for p in pathlib.Path('.').rglob(n) if '.venv' not in p.parts]"
