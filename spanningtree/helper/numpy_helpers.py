'''
Created on Mar 8, 2013

@author: max
'''
from spanningtree import np

# numpy defaults for numpy.allclose()
RTOL = 1e-05
ATOL = 1e-08


def np_allclose(a, b):
    return np.allclose(a, b, RTOL, ATOL)


def np_assert_allclose(a, b):
    return np.testing.assert_allclose(a, b, RTOL, ATOL)


def partition(p):
    x = p[..., :-1]
    y = p[..., -1:]
    return (x, y)
