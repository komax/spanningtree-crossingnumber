'''
Created on Feb 5, 2013

@author: max
'''
import numpy as np
import math
import random

class PointSet:
    def __init__(self, n, dimension):
        self.n = n
        self.dimension = dimension
        shape = (n, dimension)
        self.points = np.random.uniform(0,n, shape)
        
    def has_point(self, p):
        for point in self.points:
            if (point == p).all():
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
        return self.as_tuple()
        
    def has_edge(self, i, j):
        return self.adj_matrix[i,j] or self.adj_matrix[j,i]
    
    def update(self, i, j, new_val):
        self.adj_matrix[i,j] = self.adj_matrix[j,i] = new_val
        
def create_uniform_graph(n,d):
    return HighDimGraph(n,d)

def create_grid_graph(n,d):
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
        
    def get_lines(self):
        return self.lines.values()
    
    def get_points(self):
        return self.point_set.points[..., :-1]
    
    def get_y(self):
        return self.point_set.points[..., -1:]
    
    def create_stabbing_lines(self):
        for (i,j) in self.edges:
            self.__get_line(i, j)
        return
        
    def create_all_lines(self):
        # TODO update implementation for 3D
        pass
    
    def __create_line(self, p, q):
        return HighDimLine(np.array([p,q]))
    
    def __create_lines(self, p,q, eps):
        '''
        for two points p,q compute all four possible separation lines
        '''
        # TODO update it for high dimensions
        (x1, y1) = p
        (x2, y2) = q
        y_delta = math.fabs(y1 - y2)
        eps = 0.1
        delta = y_delta * eps
        pq_line_set = set()
        if x1 == x2:
            # special case if point are in a grid
            x1l = x1 - delta
            x1r = x1 + delta
            x2l = x2 - delta
            x2r = x2 + delta
            pq_line_set.add(self.__create_line((x1l,y1),(x2l,y2)))
            pq_line_set.add(self.__create_line((x1r,y1),(x2r,y2)))
            pq_line_set.add(self.__create_line((x1l,y1),(x2r,y2)))
            pq_line_set.add(self.__create_line((x1r,y1),(x2l,y2)))
        else:
            y1u = y1 + delta
            y1b = y1 - delta
            y2u = y2 + delta
            y2b = y2 - delta
            pq_line_set.add(self.__create_line((x1,y1u),(x2,y2u)))
            pq_line_set.add(self.__create_line((x1,y1b),(x2,y2b)))
            pq_line_set.add(self.__create_line((x1,y1u),(x2,y2b)))
            pq_line_set.add(self.__create_line((x1,y1b),(x2,y2u)))
        return pq_line_set
    
    def generate_lines(self, points):
        ''' compute all possible seperators (lines) on the point set. There maybe
            duplicates within this set
        '''
        lines = {}
        for p in points:
            for q in points:
                if points.index(p) < points.index(q):
                    if not lines.has_key((p,q)):
                        # create all different lines
                        pq_lines = create_lines(p,q, eps)
                        lines[p,q] = pq_lines
        line_set = []
        for pq_lines in lines.values():
            line_set = line_set + pq_lines
        return line_set
    
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
        #print X
        y = X[..., -1:]
        A = np.hstack([X[..., :-1], np.ones((X.shape[0], 1), dtype=X.dtype)])
        self.theta = np.linalg.lstsq(A, y)[0]
        
    def __key(self):
        return tuple(self.theta)

    def __eq__(a, b):
        return a.theta == b.theta

    def __hash__(self):
        return hash(self.__key())
        
    def __call__(self,x):
        paddedX = np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])
        print paddedX
        y = np.dot(paddedX, self.theta)
        #print y.shape
        #print np.allclose(y[0], np.array([1.]))
        #print "y=%s" % y
        return y.flatten()

    def __str__(self):
       return 'HighDimLine(theta=\n%s, points=\n%s)' % (self.theta,self.X)

    def __repr__(self):
        return self.__str__()
    
    def __partition(self, p):
        x = p[..., :-1]
        y = p[..., -1:]
        return (x,y)

    def is_on(self, p):
        (x,y) = self.__partition(p)
        y_line = self(x)
        return np.allclose(y_line, y)

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
        
    def __str__(self):
       return 'HighDimLineSegment(theta=\n%s, points=\n%s)' % (self.theta,self.X)

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
    if np.allclose(line.theta[..., :-1],line_seg.theta[..., :-1]):
        print "no crossing found"
        return False
    else:
        A = np.vstack(line.theta, line.seg.theta)
        b = - A[..., -1:]
        A[..., -1] = -np.ones(len(A))
        intersection_point = np.linalg.solve(A, b)
        print intersection_point
        x = intersection_point[..., :-1]
        return line_seg.is_between(x)