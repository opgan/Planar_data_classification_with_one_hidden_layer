# pylint: disable=unused-variable
from lib.data import injest

def test_injest():
    X_train, Y_train, X_test, Y_test, classes= injest("spiral_planar_dataset")
    Y = Y_train
    X = X_train
    m = Y.shape[1]
    assert m == 400
    assert X.shape == (2, 400)