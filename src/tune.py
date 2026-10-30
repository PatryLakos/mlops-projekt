import optuna
from optuna_integration.mlflow import MLflowCallback
from sklearn.model_selection import cross_val_score

from src.data import load_data, split_data
from src.train import STORAGE, STUDY_NAME, TRACKING_URI, build_model

N_TRIALS = 50


def objective(trial, X_train, y_train):
    # Przestrzen przeszukiwania z laboratorium 2 (z cwiczeniem samodzielnym 1)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
    }
    model = build_model(params)
    # Ocena na walidacji krzyzowej - zbior testowy nie bierze udzialu w strojeniu
    return cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()


def main():
    X, y = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    mlflow_callback = MLflowCallback(
        tracking_uri=TRACKING_URI,
        metric_name="cv_accuracy",
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        callbacks=[mlflow_callback],
    )

    print(f"Najlepsze parametry: {study.best_params}")
    print(f"Najlepsza trafnosc CV: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
