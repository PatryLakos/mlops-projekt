import mlflow
import mlflow.sklearn
import optuna
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier

from src.config import KONFIGURACJA, MLFLOW_URI, NAZWA_MODELU
from src.data import load_data, split_data
from src.evaluate import evaluate_f1, evaluate_model

# Wszystkie wartosci konfiguracyjne pochodza z config.yaml
STORAGE = KONFIGURACJA["strojenie"]["storage"]
STUDY_NAME = KONFIGURACJA["strojenie"]["nazwa_badania"]
TRACKING_URI = MLFLOW_URI
EKSPERYMENT = KONFIGURACJA["trening"]["eksperyment"]
NAZWA_RUNU = KONFIGURACJA["trening"]["nazwa_runu"]

DOMYSLNE_PARAMETRY = KONFIGURACJA["trening"]["domyslne_parametry"]


def get_best_params():
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
        return study.best_params
    except (KeyError, ValueError):
        print("Studium Optuny nie istnieje, uzyto parametrow domyslnych.")
        return DOMYSLNE_PARAMETRY


def build_model(params=None):
    if params is None:
        params = DOMYSLNE_PARAMETRY
    return RandomForestClassifier(random_state=42, **params)


def train_model(X_train, y_train, params=None):
    model = build_model(params)
    model.fit(X_train, y_train)
    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    params = get_best_params()
    model = train_model(X_train, y_train, params)
    accuracy = evaluate_model(model, X_test, y_test)
    f1 = evaluate_f1(model, X_test, y_test)
    print(f"Trafnosc modelu na zbiorze testowym: {accuracy:.4f}, F1 macro: {f1:.4f}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EKSPERYMENT)
    with mlflow.start_run(run_name=NAZWA_RUNU):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)
        sygnatura = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=sygnatura,
            input_example=X_train[:5],
            registered_model_name=NAZWA_MODELU,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )


if __name__ == "__main__":
    main()
