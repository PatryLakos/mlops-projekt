from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_test, y_test):
    predykcje = model.predict(X_test)
    return accuracy_score(y_test, predykcje)


def evaluate_f1(model, X_test, y_test):
    predykcje = model.predict(X_test)
    return f1_score(y_test, predykcje, average="macro")
