# Ćwiczenie samodzielne 1: porównanie wersji modelu i przepięcie aliasu champion
from mlflow import MlflowClient

MODEL_NAME = "breast_cancer"
VERSIONS = ["1", "2"]
ALIAS = "champion"

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

results = []
for version in VERSIONS:
    model_version = client.get_model_version(name=MODEL_NAME, version=version)
    run = client.get_run(model_version.run_id)
    results.append(
        {
            "version": version,
            "run_name": run.info.run_name,
            "accuracy": run.data.metrics["accuracy"],
            "f1_macro": run.data.metrics["f1_macro"],
        }
    )

print(f"Porównanie wersji modelu '{MODEL_NAME}'")
print(f"{'Wersja':<8} {'Uruchomienie':<26} {'accuracy':>10} {'f1_macro':>10}")
print("-" * 57)
for r in results:
    print(
        f"{r['version']:<8} {r['run_name']:<26} "
        f"{r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}"
    )

best = max(results, key=lambda r: r["f1_macro"])
client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=best["version"])
print(
    f"\nAlias '{ALIAS}' wskazuje teraz wersję {best['version']} "
    f"(f1_macro = {best['f1_macro']:.4f})"
)
