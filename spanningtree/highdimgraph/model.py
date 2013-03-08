'''
Created on Feb 5, 2013

@author: max
'''
import math
import random
import copy
from collections import deque
from spanningtree import np
from lines import HighDimLineSegment, HighDimLine
from spanningtree.helper.numpy_helpers import partition 
import factories
import crossing

class PointSet:
    def __init__(self, n, dimension):
        self.n = n
        self.d = dimension
        shape = (n, dimension)
        self.points = np.random.uniform(0, n, shape)
        self.name = ''
        
    def get_name(self):
        return self.name

    def __getitem__(self, i):
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

    def subset(self, indices_subset):
        indices_subset = sorted(indices_subset)
        for x in indices_subset:
            assert 0 <= x < self.n
        return self.points[indices_subset]

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

    def iter_subset(self, subset=None):
        if subset is None:
            self.as_tuple()
        else:
            ranged_subset = sorted(subset)
            for i in ranged_subset:
                for j in ranged_subset:
                    if i < j and self.adj_matrix[i, j]:
                        yield (i, j)

    def __iter__(self):
        return self.as_tuple()

    def __len__(self):
        length = 0
        for (i, j) in self.as_tuple():
            length += 1
        return length

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

class HighDimGraph:
    def __init__(self, points, edges, n, d):
        assert points.n == n
        assert points.d == d
        self.point_set = points
        self.edges = edges
        self.n = n
        self.d = d

        self.solution = factories.create_solution_edges(n)
        self.connected_components = ConnectedComponents(n)

        self.lines = []

        self.lines_registry = {}
        self.line_segments = {}
        
    def get_name(self):
        return self.point_set.get_name()

    def copy_graph(self):
        name = self.get_name()
        np_points = self.point_set.points
        copied_graph = factories.create_graph(np_points, self.n, self.d, name)
        copied_graph.lines = copy.deepcopy(self.lines)
        copied_graph.line_segments = copy.deepcopy(self.line_segments)
        crossing.new_crossing_registry()
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


    def euclidean_distance(self, i, j):
        p = self.point_set[i]
        q = self.point_set[j]
        return np.linalg.norm(p - q)


    def merge_cc(self, i, j):
        self.connected_components.merge_by_vertices(i, j)
        cc = self.connected_components.get_connected_component(i)
        for p in cc:
            for q in cc:
                if p != q:
                    self.edges.update(p, q, False)


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
                    j = j - 1
                    j, prev = bfs_list[j]
            else:
                if prev < p:
                    spanning_tree_edges.add((prev, p))
                else:
                    spanning_tree_edges.add((p, prev))
            prev = p
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

        self.solution = factories.create_solution_edges(self.n)
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
        number_of_random_lines = int(math.ceil(math.sqrt(2. * self.n)))
        n = 2 * number_of_random_lines
        points_for_lines = factories.create_uniform_points(n, self.d)
        lines = []
        for i in range(0, n, 2):
            pq = points_for_lines.subset((i, i + 1))
            line = HighDimLine(pq)
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
            (p, q) = self.point_set[i], self.point_set[j]
            pq_lines = self.__create_lines(p, q)
            lines += pq_lines
        self.lines = lines
        return

    def __create_lines(self, p, q):
        '''
        for two points p,q compute all four possible separation lines_registry
        '''
        # TODO update it for high dimensionsindices_subset
        (x1, y1) = partition(p)
        x1 = x1[0]
        (x2, y2) = partition(q)
        x2 = x2[0]
        y_delta = math.fabs(y1 - y2)
        eps = 0.1
        delta = y_delta * eps
        pq_lines = []
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

    def __partition_points_by_line(self, line, point_range):
        ''' partitioning of point set with discriminative function line
            (points above the line as tuples , points below the line as sets)
            both parts can be empty
        '''
        above_points = list()
        below_points = list()
        for i in point_range:
            p = self.point_set[i]
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

    def preprocess_lines(self, subset=None):
        ''' removes lines_registry that have same partitioning of the point set as
            equivalent ones
            lines_registry above or below the point set are also removed
        '''
        if subset is None:
            point_range = range(self.n)
        else:
            point_range = sorted(subset)
        lines_dict = {}
        for line in self.lines:
            partition_tuple = self.__partition_points_by_line(line, point_range)
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
            pq = self.point_set.subset((i, j))
            line = HighDimLine(pq)
            self.lines_registry[(i, j)] = line
        return self.lines_registry[(i, j)]

    def get_line_segment(self, i, j):
        if not (i, j) in self.line_segments:
            pq = self.point_set.subset((i, j))
            line_segment = HighDimLineSegment(pq)
            self.line_segments[(i, j)] = line_segment
        return self.line_segments[(i, j)]

    def calculate_crossing_with(self, line):
        '''
        for a given line calculate the crossing number (int) over all edges
        '''
        crossings = 0
        for (p, q) in self.solution:
            line_segment = self.get_line_segment(p, q)
            if crossing.has_crossing(line, line_segment):
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
            crossing_no = self.calculate_crossing_with(line)
            if crossing_no > max_crossing_number:
                max_crossing_number = crossing_no
        return max_crossing_number

    def crossing_number(self):
        ''' alias for maximum crossing number '''
        return self.maximum_crossing_number()


class ConnectedComponents:
    def __init__(self, n):
        self.ccs = list(set([i]) for i in range(n))
        
    def __iter__(self):
        for cc in self.ccs:
            yield cc

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