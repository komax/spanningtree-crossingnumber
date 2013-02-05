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
            
    def get(self, index):
        return self.points[index]
    
    def __iter__(self):
        return self.points.__iter__()
            
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
        self.lines = {}
        self.line_segments = {}
        
    def create_stabbing_lines(self):
        pass
        
    def create_all_lines(self):
        pass
    
    def __partition_points_by_line(self, line):
        ''' partitioning of point set with discriminative function line
            (points above the line as tuples , points below the line as sets)
            both parts can be empty
        '''
        above_points = set()
        below_points = set()
        for p in self.points:
            if line.is_on(p):
                return ()
            elif line.is_above(p):
                above_points.add(p)
            elif line.is_below(p):
                below_points.add(p)
            else:
                raise StandardError('can not find point p=%s on line=%s' %
                        (p,line))
        return (above_points, below_points)
    
    def preprocess_lines(self):
        ''' removes lines that have same partitioning of the point set as
            equivalent ones
            lines above or below the point set are also removed
        '''
        lines_dict = {}
        lines = self.lines.values()
        for line in lines:
            # FIXME update this implementation 
            partition_tuple = self.__partition_points_by_line(line)
            if not partition_tuple:
                # skip this line, because one point is on this line
                continue
            elif not partition_tuple[0] or not partition_tuple[1]:
                # above or below part is empty, skip lineis_above
                continue
            elif lines_dict.has_key(partition_tuple):
                # skip this line, there is one equivalent line stored
                continue
            else:
                # new equivalent class, store this line
                lines_dict[partition_tuple] = line
        self.lines = lines_dict
    
    def __get_line(self, p,q):
        if not (p,q) in self.lines:
            X = np.array([self.point_set.get(p), self.point_set.get(q)])
            line = HighDimLine(X)
            self.lines[(p,q)] = line
        return self.lines[(p,q)]
    
    def __get_line_segment(self, p,q):
        if not (p,q) in self.line_segments:
            X = np.array([self.point_set.get(p), self.point_set.get(q)])
            line_segment = HighDimLineSegment(X)
            self.line_segments[(p,q)] = line_segment
        return self.lines[(p,q)]

    def calculate_crossing_with(self, line):
        '''
        for a given line calculate the crossing number (int) over all edges
        '''
        crossings = 0
        for (p,q) in self.edges:
            line_segment = self.__get_line_segment(p,q)
            if has_crossing(line, line_segment):
                crossings += 1
        return crossings
    
    def crossing_tuple(self, solution):
        '''
        returns (crossing number, overall crossings)
        on all lines with edges
        '''
        crossings = 0
        max_crossing_number = 0
        for line in self.lines.values():
            crossing_number = calculate_crossing_with(line, solution)
            crossings += crossing_number
            if crossing_number > max_crossing_number:
                max_crossing_number = crossing_number
        return (max_crossing_number, crossings)

    def calculate_crossings(self, solution):
        '''
        for all lines and edges in the solution compute the overall crossings
        '''
        crossing_number = 0
        for line in self.lines.values():
            crossing_number += calculate_crossing_with(line, solution)
        return crossing_number

    def maximum_crossing_number(lines, solution):
        '''
        for all lines and edges in the solution compute the maximum crossing number
        '''
        max_crossing_number = 0
        for line in lines:
            crossing_number = calculate_crossing_with(line, solution)
        if crossing_number > max_crossing_number:
            max_crossing_number = crossing_number
        return max_crossing_number

    def crossing_number(self):
        ''' alias for maximum crossing number '''
        return self.maximum_crossing_number()
        
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