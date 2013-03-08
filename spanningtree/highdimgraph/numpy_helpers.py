'''
Created on Mar 8, 2013

@author: max
'''
# numpy defaults for numpy.allclose()
RTOL = 1e-05
ATOL = 1e-08

def np_allclose(a, b):
    return np.allclose(a, b, RTOL, ATOL)

def np_assert_allclose(a, b):
    return np.testing.assert_allclose(a, b, RTOL, ATOL)