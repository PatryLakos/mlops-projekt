# Laboratorium 3 - serwer MLflow w Dockerze (Docker Desktop musi byc uruchomiony)
docker pull ghcr.io/mlflow/mlflow:v3.14.0

docker run -d `
  --name mlflow-server `
  -p 5000:5000 `
  -v mlflow-data:/mlflow `
  ghcr.io/mlflow/mlflow:v3.14.0 `
  mlflow server `
  --host 0.0.0.0 `
  --port 5000 `
  --backend-store-uri sqlite:////mlflow/mlflow.db `
  --artifacts-destination /mlflow/artifacts `
  --cors-allowed-origins "*"

docker ps
# UI: http://localhost:5000
# Zatrzymanie / ponowne uruchomienie:
#   docker stop mlflow-server
#   docker start mlflow-server

# Kolejnosc skryptow lab 3 (z katalogu glownego repo):
#   python src/train_tracked.py
#   python src/tune_tracked.py
#   python src/register_model.py        # wpisz best_params z tune_tracked
#   python src/set_alias.py
#   python src/predict_from_registry.py
#   python src/register_model_v2.py     # cw. samodzielne 1
#   python src/compare_versions.py      # cw. samodzielne 1
#   python src/tune_nested.py           # cw. samodzielne 2
#   git tag lab-03-done ; git push --tags
