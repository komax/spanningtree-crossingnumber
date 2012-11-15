'''
computes a spanning tree for a point set s.t. it has low crossing number to
the line set, using the multiplicative weights method
'''

from datagenerator import Line2D, LineSegment2D

def preprocess_lines(lines):
    # TODO remove unnecessary lines
    return lines

'''
This class holds all functionality of a graph between different connected
components. Some boolean methods for checking connectivity, adjacent of
vertices
'''
class Graph:
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
        else:
            return False

    def merge_connected_components(self, c, oc):
        assert c in self.connected_components
        assert oc in self.connected_components
        assert c != oc
        self.connected_components.remove(c)
        self.connected_components.remove(oc)
        for p in c:
            for q in oc:
                self.adjacent_list[p].remove(q)
                self.adjacent_list[q].remove(p)
        new_connected_component = c + oc
        self.connected_components.append(new_connected_component)
        return

    def merge_cc_with_vertics(self, u, v):
        assert u != v
        try:
            c1 = self.get_connected_component(u)
            c2 = self.get_connected_component(v)
            self.merge_connected_components(c1, c2)
            return
        except:
            raise


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
                if not (v,u) in edges:
                    edges.append((u,v))
        return edges

    def get_connected_component(self, u):
        for c in self.connected_components:
            if u in c:
                return c
        else:
            raise StandardError('can not find vertex=%s in this graph' % u)


def create_graph(points):
    connected_components = []
    for p in points:
        connected_components.append([p])
    return Graph(connected_components)

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
        print edge
        line_segment = edge_to_linesegment(edge)
        weights[edge] = 0.0
        for line in lines:
            if has_crossing(line, line_segment):
                weights[edge] += line_weights[line]
    #print weights
    #for edge in weights:
    #    print "%s => %s" % (edge, weights[edge])
    min_edge = min(weights, key=weights.get)
    #print min_edge
    return min_edge


def compute_spanning_tree(points, lines):
    # TODO checkin implementation
    lines = preprocess_lines(lines)
    solution = []
    number_of_crossings = {}
    weights = {}
    graph = create_graph(points)
    while len(points) > 1:
        for line in lines:
            number_of_crossings[line] = calculate_crossing_with(line, solution)
            weights[line] = 2**(number_of_crossings[line])
        #print "line weights = %s" % weights
        min_edge = find_min_edge(graph.get_edges(), lines, weights)
        (p,q) = min_edge
        graph.merge_cc_with_vertics(p,q)
        print "points = %s, point p=%s, q=%s" % (points,p,q)
        if p in points:
            points.remove(p)
        elif q in points:
            points.remove(q)
        else:
            raise StandardError()
        solution.append(min_edge)
    return solution

