# tests/unit/test_models.py
import pytest
from src.models.xgboost_model import XGBoostModel

@pytest.fixture
def sample_data():
    # Generar datos sintéticos
    ...

def test_model_training(sample_data):
    X_train, y_train = sample_data
    model = XGBoostModel(config)
    model.fit(X_train, y_train)
    assert model.model is not None

def test_walk_forward_validation(sample_data):
    X, y = sample_data
    model = XGBoostModel(config)
    predictions, actuals = model.walk_forward_validation(X, y)
    assert len(predictions) == len(actuals)
