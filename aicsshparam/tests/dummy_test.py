import numpy as np


def test_dummy():
    a = np.ones((5, 5))
    a[0, 0] = 2

    assert np.any(a > 0)
