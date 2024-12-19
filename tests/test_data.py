import myLib.data

def test_injest():
    X, Y = myLib.data.injest()
    m = Y.shape[1]
    assert m == 400
    assert X.shape == (2, 400)