import numpy as np
import pytest

from src.data import load_data, split_data
from src.evaluate import evaluate_model
from src.train import train_model


def test_load_data():
    """Test danych: wymiary, kompletnosc i liczba klas."""
    X, y = load_data()
    # Liczba obserwacji w X i y musi byc zgodna
    assert X.shape[0] == y.shape[0]
    # Zbior nie moze zawierac wartosci pustych
    assert not np.isnan(X).any()
    # Problem jest klasyfikacja binarna, oczekiwane sa dwie klasy
    assert len(np.unique(y)) == 2


def test_ksztalt_predykcji():
    """Test kontraktu: ksztalt i dziedzina predykcji."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    predykcje = model.predict(X_test)
    # Liczba predykcji musi odpowiadac liczbie obserwacji testowych
    assert predykcje.shape[0] == X_test.shape[0]
    # Predykcje moga przyjmowac wylacznie wartosci etykiet klas
    assert set(np.unique(predykcje)).issubset({0, 1})


def test_prog_trafnosci():
    """Test walidacyjny: minimalna akceptowalna jakosc modelu."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    # Model gorszy od progu oznacza regresje jakosci w projekcie
    assert accuracy > 0.90


@pytest.mark.parametrize("test_size", [0.2, 0.3])
def test_split_data(test_size):
    """Cwiczenie samodzielne 1.1: licznosci i proporcja podzialu."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    # Zadna obserwacja nie ginie i nie jest duplikowana
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # Rozmiar zbioru testowego zgodny z test_size (scikit-learn zaokragla w gore)
    assert X_test.shape[0] == int(np.ceil(test_size * X.shape[0]))


def test_stratyfikacja_i_odtwarzalnosc():
    """Cwiczenie samodzielne 1.2: proporcje klas po podziale i powtarzalnosc."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Podzial stratyfikowany zachowuje udzial klasy pozytywnej (tolerancja 1 p.p.)
    assert abs(y_train.mean() - y.mean()) < 0.01
    assert abs(y_test.mean() - y.mean()) < 0.01
    # Ustalone ziarno losowosci daje identyczne predykcje w dwoch treningach
    pred_1 = train_model(X_train, y_train).predict(X_test)
    pred_2 = train_model(X_train, y_train).predict(X_test)
    assert np.array_equal(pred_1, pred_2)
