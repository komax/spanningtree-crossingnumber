'''
Created on Feb 5, 2013

@author: max
'''
import numpy as np

class PointSet:
    def __init__(self, n, dimension):
        self.n = n
        self.dimension = dimension
        shape = (n, dimension)
        self.points = np.random.uniform(0,n, shape)
        
    def has_point(self, p):
        for row in range(0,self.n):
            if (self.points[row] == p).all():
                return True
        else:
            return False
        
    def get_index(self, p):
        for row in range(0,self.n):
            if (self.points[row] == p).all():
                return row
            
    def subset(self, subset_points):
        indices_subset = set()
        for point in subset_points:
            if self.has_point(point):
                indices_subset.add(self.get_index(point))
        return indices_subset
            
class Edges:
    def __init__(self, n):
        self.n = n
        self.adj_matrix = np.ones((n,n), dtype=bool)
        for i in range(0,n):
            self.adj_matrix[i,i] = False
            
    def as_tuple(self):
        for i in range(0, self.n):
            for j in range(0, self.n):
                if i < j and self.adj_matrix[i,j]:
                    yield (i,j)
            
    def __iter__(self):
        self.as_tuple()
        
def create_uniform_graph(n,d):
    return HighDimGraph(n,d)

def create_grid_point_graph(n,d):
    assert d == 2
    point_set = PointSet(n, d)
    root_n = int(math.ceil(math.sqrt(n)))
    eps = 0.1
    x = 0.0
    y = 0.0
    row = 0
    for i in range(root_n):
        y = 0.0
        for j in range(root_n):
            x_eps = random.uniform(-eps,eps)
            y_eps = random.uniform(-eps,eps)
            point_set.points[row] = (x+x_eps,y+y_eps)
            row += 1
            y += 5.0
        x += 5.0
    assert row == n
    graph = HighDimGraph(n,d)
    graph.point_set = point_set
    return graph
        
class HighDimGraph:
    def __init__(self, n, d):
        self.point_set = PointSet(n, d)
        self.edges = Edges(n)
        
        
class HighDimLine:
    def __init__(self, X):
        self.X = X
        y = X[..., -1:]
        A = np.vstack([X[..., :-1], np.ones(len(X))]).T
        self.theta = np.linalg.lstsq(A, y)[0]
        
    def __key(self):
        return tuple(self.theta)

    def __eq__(a, b):
        return a.theta == b.theta

    def __hash__(self):
        return hash(self.__key())
        
    def __call__(self,x):
        paddedX = np.vstack([x, np.ones(len(x))])
        y = np.dot(theta, paddedX.T)
        return y

    def __str__(self):
       return 'HighDimLine(theta=%s, points=%s)' % (self.theta,self.X)

    def __repr__(self):
        return self.__str__()
    
    def __partition(self, p):
        x = p[..., :-1]
        y = p[..., -1:]
        return (x,y)

    def is_on(self, p):
        (x,y) = self.__partition(p)
        y_line = self(x)
        return np.allclose(yline, y)

    def is_above(self, p):
        (x,y) = self.__partition(p)
        y_line = self(x)
        return (y > y_line).all() and not self.is_on(p)

    def is_below(self, p):
        (x,y) = self.__partition(p)
        y_line = self(x)
        return (y < y_line).all() and not self.is_on(p)
    
class HighDimLineSegment(HighDimLine):
    def __init__(self, X):
        HighDimLine.__init__(self, X)

    def is_between(self, x):
        res = np.cross(X[0]-x, X[1]-x)
        return np.allclose(res, 0.0)

    def is_on(self, p):
        (x,y) = self.__partition(p)
        if self.is_between(x):
            return HighDimLine.is_on(self, p)
        else:
            return False

    def __call__(self, x):
        if self.is_between(x):
            return HighDimLine.__call__(self, x)
        
def has_crossing(line, line_seg):
    '''
    Has line a crossing with the line segment
    '''
    if np.allclose(line.theta,line_seg.theta):
        return False
    else:
        A = np.array([line.theta, line.seg.theta])
        b = - A[..., -1:]
        A[..., -1] = -np.ones(len(A))
        intersection_point = np.linalg.solve(A, b)
        x = intersection_point[..., :-1]
        return line_seg.is_between(x)
    
