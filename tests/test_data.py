from lib.data import injest

def test_injest():
    X, Y = injest()
    m = Y.shape[1]
    assert m == 400
    assert X.shape == (2, 400)