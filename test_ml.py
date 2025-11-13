import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    """
    # Your code here
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict"), "Model should have a predict method"


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)
    model = train_model(np.random.rand(100, 5), np.random.randint(0, 2, 100))
    preds = inference(model, X_test)
    
    assert len(preds) == len(y_test), "Number of predictions should match number of samples."
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions should be binary (0 or 1)."


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    y_true = np.array([0, 1, 1, 0, 1, 0])
    preds = np.array([0, 1, 0, 0, 1, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, preds)
    
    for metric in (precision, recall, fbeta):
        assert 0.0 <= metric <= 1.0, "Metric values should be between 0 and 1."
