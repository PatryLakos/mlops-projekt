from mlflow import MlflowClient

MODEL_NAME = "breast_cancer"

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=1,
)
print(f"Alias 'champion' -> {MODEL_NAME} v1")
