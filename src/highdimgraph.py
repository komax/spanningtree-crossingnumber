'''
Created on Feb 5, 2013

@author: max
'''
import numpy as np
import math
import random
import copy
from collections import deque
from pydoc import deque

# numpy defaults for numpy.allclose()
RTOL = 1e-05
ATOL = 1e-08

def np_allclose(a, b):
    return np.allclose(a, b, RTOL, ATOL)

def np_assert_allclose(a, b):
    return np.testing.assert_allclose(a, b, RTOL, ATOL)

class PointSet:
    def __init__(self, n, dimension):
        self.n = n
        self.d = dimension
        shape = (n, dimension)
        self.points = np.random.uniform(0, n, shape)
        
    def get_point(self, i):
        return self.points[i]
        
    def has_point(self, p):
        for point in self.points:
            if np_allclose(point, p):
                return True
        else:
            return False
        
    def get_index(self, p):
        for row in range(0, self.n):
            if np_allclose(self.points[row], p):
                return row
    
    def __iter__(self):
        return self.points.__iter__()
    
    def __str__(self):
        return 'PointSet(n=%s,d=%s,points=%s' % (self.n, self.d, self.points)
    
    def __repr__(self):
        return self.__str__()
            
    def subset(self, subset_points):
        indices_subset = set()
        for point in subset_points:
            if self.has_point(point):
                indices_subset.add(self.get_index(point))
        return indices_subset
    
def create_uniform_points(n, d):
    return PointSet(n, d)

def create_grid_points(n, d):
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
            x_eps = random.uniform(-eps, eps)
            y_eps = random.uniform(-eps, eps)
            point_set.points[row] = np.array((x + x_eps, y + y_eps))
            row += 1
            y += 5.0
        x += 5.0
    assert row == n
    return point_set

def create_pointset(np_array, n, d):
    assert np_array.shape == (n, d)
    point_set = PointSet(n, d)
    point_set.points = np_array
    return point_set
            
class Edges:
    def __init__(self, n, matrix):
        self.n = n
        assert matrix.shape == (n, n)
        assert matrix.dtype == bool
        self.adj_matrix = matrix
            
    def as_tuple(self):
        for i in range(0, self.n):
            for j in range(0, self.n):
                if i < j and self.adj_matrix[i, j]:
                    yield (i, j)
            
    def __iter__(self):
        return self.as_tuple()
    
    def __str__(self):
        return 'Edges(n=%s, adj_matrix=%s)' % (self.n, self.adj_matrix)
    
    def __repr__(self):
        return self.__str__()
        
    def has_edge(self, i, j):
        return self.adj_matrix[i, j] or self.adj_matrix[j, i]
    
    def adj_nodes(self, i):
        neighbors = set()
        for j in range(self.n):
            if i != j and self.adj_matrix[i, j]:
                neighbors.add(j)
        return neighbors
    
    def adj_edges(self, i):
        for j in self.adj_nodes(i):
            yield (i, j)
    
    def update(self, i, j, new_val):
        self.adj_matrix[i, j] = self.adj_matrix[j, i] = new_val
        
def create_all_edges(n):
    adj_matrix = np.ones((n, n), dtype=bool)
    for i in range(0, n):
        adj_matrix[i, i] = False
    edges = Edges(n, adj_matrix)
    return edges
        
def create_solution_edges(n):
    sol_matrix = np.zeros((n, n), dtype=bool)
    solution = Edges(n, sol_matrix)
    return solution
        
def create_uniform_graph(n, d):
    points = create_uniform_points(n, d)
    edges = create_all_edges(n)
    return HighDimGraph(points, edges, n, d)

def create_grid_graph(n, d):
    points = create_grid_points(n, d)
    edges = create_all_edges(n)
    return HighDimGraph(points, edges, n, d)

def create_graph(points, n, d):
    point_set = create_pointset(points, n, d)
    edges = create_all_edges(n)
    return HighDimGraph(point_set, edges, n, d)

class ConnectedComponents:
    def __init__(self, n):
        self.ccs = list(set([i]) for i in range(n))
        
    def get_connected_component(self, i):
        for cc in self.ccs:
            if i in cc:
                return cc
        else:
            raise StandardError('can not find vertex=%s in this connected components=%s' % (i, self.ccs))
    
    def merge(self, cc1, cc2):
        ''' 
        merge of two different connected components to a new one
        '''
        assert cc1 in self.ccs
        assert cc2 in self.ccs
        if cc1 != cc2:
            self.ccs.remove(cc2)
            cc1.update(cc2)
        return
    
    def merge_by_vertices(self, i, j):
        ''' 
        merge of two connected components but parameters are two points
        from the corresponding connected components
        '''
        try:
            cc1 = self.get_connected_component(i)
            cc2 = self.get_connected_component(j)
            return self.merge(cc1, cc2)
        except:
            raise
        
    def __len__(self):
        return len(self.ccs)
        
class HighDimGraph:
    def __init__(self, points, edges, n, d):
        assert points.n == n
        assert points.d == d
        self.point_set = points
        self.edges = edges
        self.n = n
        self.d = d
        
        self.solution = create_solution_edges(n)
        self.connected_components = ConnectedComponents(n)
        
        self.lines = []
        
        self.lines_registry = {}
        self.line_segments = {}
        
    def copy_graph(self):
        np_points = self.point_set.points
        copied_graph = create_graph(np_points, self.n, self.d)
        copied_graph.lines = copy.deepcopy(self.lines)
        return copied_graph
    
    def bfs(self, root):
        visited = set([root])
        queue = deque([root])
        
        while queue:
            i = queue.popleft()
            visited.add(i)
            yield i
            for neighbor in self.solution.adj_nodes(i):
                if neighbor not in visited:
                    queue.append(neighbor)
                    
    def compute_connected_component(self, root):
        connected_component = set([root])
        for bfs_node in self.bfs(root):
            connected_component.add(bfs_node)
        return connected_component
    
    def compute_connected_components(self):
        remaining_points = range(self.n)
        while remaining_points:
            p = remaining_points.pop()
            connected_component = self.compute_connected_component(p)
            for cc_p in connected_component:
                self.connected_components.merge_by_vertices(p, cc_p)
                if cc_p in remaining_points:
                    remaining_points.remove(cc_p)      
        return
                    
    def spanning_tree(self, root):
        spanning_tree_edges = set()
        bfs_list = list(enumerate(self.bfs(root)))
        prev = root
        for (i, p) in bfs_list:
            j = i - 1
            # find first edge upwards from nearest neighbor to root
            while not self.solution.has_edge(p, prev):
                # do not go up if p == root
                if i == 0:
                    break
                else:
                    j = j-1
                    j, prev = bfs_list[j]
            else:
                if prev < p:
                    spanning_tree_edges.add((prev, p))
                else:
                    spanning_tree_edges.add((p, prev))
            prev = p  
#        print "spanning tree_edges from root=%s: >%s<" % (root, spanning_tree_edges)
        return spanning_tree_edges
    
    def compute_spanning_tree_on_ccs(self):
        new_solution_edges = set()
        remaining_points = range(self.n)
        while remaining_points:
            p = remaining_points.pop()
            spanning_tree_edges = self.spanning_tree(p)
            new_solution_edges.update(spanning_tree_edges)
            # TODO maybe use bfs with ccs instead to delete from remain.points
            for (i, j) in spanning_tree_edges:
                if i in remaining_points:
                    remaining_points.remove(i)
                if j in remaining_points:
                    remaining_points.remove(j)
                    
        self.solution = create_solution_edges(self.n)
        for (i, j) in new_solution_edges:
            self.solution.update(i, j, True)
        return 
        
    def create_stabbing_lines(self):
        lines = []
        for (i, j) in self.edges:
            line = self.__get_line(i, j)
            lines.append(line)
        self.lines = lines
        return
    
    def create_random_lines(self):
        n = 2 * self.n
        points_for_lines = create_uniform_points(n, self.d)
        lines = []
        for i in range(0,n,2):
            p = points_for_lines.get_point(i)
            q = points_for_lines.get_point(i + 1)
            line = create_line(p, q)
            lines.append(line)
        self.lines = lines
        return
    
    
    def create_all_lines(self):
        ''' 
        compute all possible seperators (lines_registry) on the point set. There maybe
        duplicates within this set
        '''
        # TODO update implementation for 3D
        lines = []
        for (i, j) in self.edges:
            (p, q) = self.point_set.get_point(i), self.point_set.get_point(j)
            pq_lines = self.__create_lines(p, q)
            lines += pq_lines
#        print lines
        self.lines = lines
        return
    
    def __create_lines(self, p, q):
        '''
        for two points p,q compute all four possible separation lines_registry
        '''
        # TODO update it for high dimensions
        (x1, y1) = partition(p)
        x1 = x1[0]
#        print x1
#        print y1
        (x2, y2) = partition(q)
        x2 = x2[0]
#        print x2
#        print y2
        y_delta = math.fabs(y1 - y2)
#        print y_delta
        eps = 0.1
        delta = y_delta * eps
        pq_lines = []
        if x1 == x2:
            # special case if point are in a grid
            x1l = x1 - delta
            x1r = x1 + delta
            x2l = x2 - delta
            x2r = x2 + delta
            pq_lines.append(HighDimLine(np.array([(x1l, y1), (x2l, y2)])))
            pq_lines.append(HighDimLine(np.array([(x1r, y1), (x2r, y2)])))
            pq_lines.append(HighDimLine(np.array([(x1l, y1), (x2r, y2)])))
            pq_lines.append(HighDimLine(np.array([(x1r, y1), (x2l, y2)])))
        else:
            y1u = y1 + delta
            y1b = y1 - delta
            y2u = y2 + delta
            y2b = y2 - delta
            pq_lines.append(HighDimLine(np.array([(x1, y1u), (x2, y2u)])))
            pq_lines.append(HighDimLine(np.array([(x1, y1b), (x2, y2b)])))
            pq_lines.append(HighDimLine(np.array([(x1, y1u), (x2, y2b)])))
            pq_lines.append(HighDimLine(np.array([(x1, y1b), (x2, y2u)])))
        return pq_lines
    
    def generate_lines(self, points):
        ''' compute all possible seperators (lines_registry) on the point set. There maybe
            duplicates within this set
        '''
        lines = {}
        for p in points:
            for q in points:
                if points.index(p) < points.index(q):
                    if not lines.has_key((p, q)):
                        # create all different lines_registry
                        pq_lines = create_lines(p, q, eps)
                        lines[p, q] = pq_lines
        line_set = []
        for pq_lines in lines.values():
            line_set = line_set + pq_lines
        return line_set
    
    def __partition_points_by_line(self, line):
        ''' partitioning of point set with discriminative function line
            (points above the line as tuples , points below the line as sets)
            both parts can be empty
        '''
        above_points = list()
        below_points = list()
        for i in range(self.n):
            # print self.point_set[i]
            p = self.point_set.get_point(i)
            if line.is_on(p):
                return ()
            elif line.is_above(p):
                above_points.append(i)
            elif line.is_below(p):
                below_points.append(i)
            else:
                raise StandardError('can not find point i=%s:p=%s on line=%s' % 
                        (i, p, line))
        above_points.sort()
        below_points.sort()
        return (tuple(above_points), tuple(below_points))
    
    def preprocess_lines(self):
        ''' removes lines_registry that have same partitioning of the point set as
            equivalent ones
            lines_registry above or below the point set are also removed
        '''
        lines_dict = {}
        for line in self.lines:
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
        self.lines = lines_dict.values()
    
    def __get_line(self, i, j):
        if not (i, j) in self.lines_registry:
            (p, q) = self.point_set.get_point(i), self.point_set.get_point(j)
            line = create_line(p, q)
            self.lines_registry[(i, j)] = line
        return self.lines_registry[(i, j)]
    
    def get_line_segment(self, i, j):
        if not (i, j) in self.line_segments:
            (p, q) = self.point_set.get_point(i), self.point_set.get_point(j)
            line_segment = create_linesegment(p, q)
            self.line_segments[(i, j)] = line_segment
        return self.line_segments[(i, j)]

    def calculate_crossing_with(self, line):
        '''
        for a given line calculate the crossing number (int) over all edges
        '''
        crossings = 0
        for (p, q) in self.solution:
            line_segment = self.get_line_segment(p, q)
            if has_crossing(line, line_segment):
                crossings += 1
        return crossings
    
    def crossing_tuple(self):
        '''
        returns (crossing number, overall crossings)
        on all lines_registry with edges
        '''
        crossings = 0
        max_crossing_number = 0
        for line in self.lines:
            crossing_number = self.calculate_crossing_with(line)
            crossings += crossing_number
            if crossing_number > max_crossing_number:
                max_crossing_number = crossing_number
        return (max_crossing_number, crossings)

    def calculate_crossings(self):
        '''
        for all lines_registry and edges in the solution compute the overall crossings
        '''
        crossing_number = 0
        for line in self.lines:
            crossing_number += self.calculate_crossing_with(line)
        return crossing_number

    def maximum_crossing_number(self):
        '''
        for all lines_registry and edges in the solution compute the maximum crossing number
        '''
        max_crossing_number = 0
        for line in self.lines:
            crossing_number = self.calculate_crossing_with(line)
        if crossing_number > max_crossing_number:
            max_crossing_number = crossing_number
        return max_crossing_number

    def crossing_number(self):
        ''' alias for maximum crossing number '''
        return self.maximum_crossing_number()
    
def create_line(p, q):
    assert len(p.shape) == 1
    assert len(q.shape) == 1
    X = np.array([p, q])
    return HighDimLine(X)
        
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
        # self.theta = self.theta.flatten()
        # print "theta=%s, shape of it %s" %(self.theta, self.theta.shape)
        
    def __key(self):
        return tuple(self.theta)

    def __eq__(a, b):
        return np_allclose(a.theta, b.theta)

    def __hash__(self):
        return hash(self.__key())
    
    def call(self, p):
        # print p
        # print p.shape, len(p.shape)
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


def partition(p):
    x = p[..., :-1]
    y = p[..., -1:]
    return (x, y)
    
def create_linesegment(p, q):
    assert len(p.shape) == 1
    assert len(q.shape) == 1
    X = np.array([p, q])
    return HighDimLineSegment(X)        
    
class HighDimLineSegment(HighDimLine):
    def __init__(self, X):
        HighDimLine.__init__(self, X)
        
    def __str__(self):
       return 'HighDimLineSegment(theta=\n%s, points=\n%s)' % (self.theta, self.X)

    def is_between(self, x):
        # TODO update this implementation for higher d. Is this still correct?
        # p = self.X[0, ..., :-1]
        # q = self.X[1, ..., :-1]
        p = self.X[0, 0, ...]
        q = self.X[1, 0, ...]
        if p > q:
            (p, q) = (q, p)
        # FIXME: why this is not working
        # print p, q, x
        # print (p <= x <= q).all()
        # print q >= x
        # print x >= p
        return (p <= x <= q).all()
        # if True:   
        #    print "in is_between: %s <= %s <= %s == %s" % (p, x, q, res) 
        #    return res
        # else:
        #    return False
        # res = np.abs(np.cross(p-x, q-x))/np.abs(p-x)
        # print "in is_between: res=%s, allclose=%s" % (res, np_allclose(res, 0.0))
        # return np_allclose(res, 0.0)

    def is_on(self, p):
        (x, y) = partition(p)
        if self.is_between(x):
            return HighDimLine.is_on(self, p)
        else:
            return False

    def __call__(self, X):
        if self.is_between(X):
            return HighDimLine.__call__(self, X)
        
def has_crossing(line, line_seg):
    '''
    Has line a crossing with the line segment
    '''
    
    # print line.theta, line_seg.theta
    # print line.theta[..., :-1], line_seg.theta[..., :]
    if np_allclose(line.theta[..., :-1], line_seg.theta[..., :-1]):
        # print "no crossing found"
        return False
    else:
        A = np.vstack([line.theta, line_seg.theta])
        b = -A[..., -1:]
        A[..., -1] = -np.ones(len(A))
        intersection_point = (np.linalg.solve(A, b)).flatten() 
        # print "intersection point= %s" % intersection_point
        x = intersection_point[..., :-1]
        # print x
        return line_seg.is_between(x)
