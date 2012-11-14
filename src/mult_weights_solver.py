'''
computes a spanning tree for a point set s.t. it has low crossing number to
the line set, using the multiplicative weights method
'''

from datagenerator import Line2D, LineSegment2D

def preprocess_lines(lines):
    # TODO remove unnecessary lines
    return lines

'''
This class holds all edges between different connected components. Some boolean
methods for checking connectivity, adjacent of vertices
'''
class Edges:
    def __init__(self, connected_components):
        assert connected_components
        self.vertices = []
        self.connected_components = connected_components
        self.adjacent_list = {}
        for c in connected_components:
            assert c
            self.vertices += c
            for p in c:
                self.adjacent_list[p] = []
                for other_c in connected_components:
                    if other_c != c:
                        assert not p in other_c
                        self.adjacent_list[p] += other_c

    def in_same_connected_component(self, u, v):
        for c in self.connected_components:
            if u in c:
                return v in c
            elif v in c:
                return u in c
        return False

    def is_adjacent(self, u, v):
        if self.adjacent_list.has_key(u):
            return v in self.adjacent_list[u]
        elif self.adjacent_list.has_key(v):
            return u in self.adjacent_list[v]
        else:
            return False

    def get_adjacent_vertices(self, u):
        if self.adjacent_list.has_key(u):
            return self.adjacent_list[u]
        else:
            return []

    def get_edges(self):
        edges = []
        for u in self.adjacent_list:
            for v in self.adjacent_list[u]:
                edges.append((u,v))
        return edges

def generate_edges(points):
    connected_components = []
    for p in points:
        connected_components.append([p])
    return Edges(connected_components)

def edge_to_linesegment(edge):
    p, q = edge
    return LineSegment2D(p, q)

def has_crossing(line, line_seg):
    if line.slope == line_seg.slope:
        return False
    else:
        x_s = (line.y_intercept - line_seg.y_intercept) / (line_seg.slope -
                line.slope)
        y_s = line(x_s)
        return line_seg.is_on((x_s, y_s))

def calculate_crossing_with(line, edges):
    crossings = 0
    for edge in edges:
        line_segment = edge_to_linesegment(edge)
        if has_crossing(line, line_segment):
            crossings += 1
    return crossings

def bfs(vertex, points, edges):
    queue = [vertex]
    visited = [vertex]
    while queue:
        u = queue.pop(0)
        for (not_used, v) in edges.select(u, "*"):
            if not v in visited:
                visited.append(v)
                queue.append(v)
        for (v, not_used) in edges.select("*", u):
            if not v in visited:
                visited.append(v)
                queue.append(v)
    return visited


def connected_components(points, edges):
    # TODO compute connected components
    pass

def edges_crossing_connected_components(c_components):
    # TODO update implementation
    edges = []
    for connected_component in c_components:
        for other_c_component in c_components:
            if (c_components.index(connected_components) <
            c_components.index(other_c_component)):
                for p in connected_component:
                    edges = edges + generate_edges([p]+c_components)
    return edges


def find_min_edge(selected_edges, lines, line_weights):
    weights = {}
    for edge in selected_edges:
        line_segment = edge_to_linesegment(edge)
        weights[edge] = 0.0
        for line in lines:
            if has_crossing(line, line_segment):
                weights[edges] += line_weights[line]
    min_edge = min(weights, weights.get)
    return min_edge


def compute_spanning_tree(points, lines):
    lines = preprocess_lines(lines)
    edges = []
    number_of_crossings = {}
    weights = {}
    while len(points) > 1:
        for line in lines:
            number_of_crossings[line] = calculate_crossing_with(line, edges)
            weights[line] = 2**(number_of_crossings[line])
        c_components = connected_components(points, edges)
        edges_between_c_components = \
                edges_crossing_connected_components(c_components)
        min_edge = find_min_edge(edges_between_c_components, lines, weights)
        (p,q) = min_edge
        points.remove(p)
        edges.append(min_edge)
    return edges

