'''
Created on Mar 8, 2013

@author: max
'''
from spanningtree import np
from spanningtree.helper.numpy_helpers import partition, np_allclose

class HighDimLine:
    def __init__(self, X):
        assert X.shape[0] == 2
        p = X[0]
        q = X[1]
        if p[..., -1] < q[..., -1]:
            self.X = X
        else:
            self.X = np.array([q, p])
        y = X[..., -1:]
        A = np.hstack([X[..., :-1], np.ones((X.shape[0], 1), dtype=X.dtype)])
        self.theta = (np.linalg.lstsq(A, y)[0]).flatten()

    def __key(self):
        return tuple(self.theta)

    def __eq__(a, b):
        return np_allclose(a.theta, b.theta)

    def __hash__(self):
        return hash(self.__key())

    def call(self, p):
        assert len(p.shape) == 1
        X = np.array([p])
        y = self(X)
        return y[0]

    def __call__(self, X):
        paddedX = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        y = np.dot(paddedX, self.theta)
        return y.flatten()

    def __str__(self):
       return 'HighDimLine(theta=\n%s, points=\n%s)' % (self.theta, self.X)

    def __repr__(self):
        return self.__str__()

    def is_on(self, p):
        (x, y) = partition(p)
        y_line = self.call(x)
        return np_allclose(y_line, y)

    def is_above(self, p):
        (x, y) = partition(p)
        y_line = self.call(x)
        return (y > y_line).all() and not self.is_on(p)

    def is_below(self, p):
        (x, y) = partition(p)
        y_line = self.call(x)
        return (y < y_line).all() and not self.is_on(p)
    
class HighDimLineSegment(HighDimLine):
    def __init__(self, X):
        HighDimLine.__init__(self, X)

    def __str__(self):
       return 'HighDimLineSegment(theta=\n%s, points=\n%s)' % (self.theta, self.X)

    def is_between(self, x):
        # TODO update this implementation for higher d. Is this still correct?
        p = self.X[0, 0, ...]
        q = self.X[1, 0, ...]
        if p > q:
            (p, q) = (q, p)
        return (p <= x <= q).all()

    def is_on(self, p):
        (x, y) = partition(p)
        if self.is_between(x):
            return HighDimLine.is_on(self, p)
        else:
            return False

    def __call__(self, X):
        if self.is_between(X):
            return HighDimLine.__call__(self, X)
