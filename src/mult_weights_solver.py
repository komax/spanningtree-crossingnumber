'''
computes a spanning tree for a point set s.t. it has low crossing number to
the line set, using the multiplicative weights method
'''
import copy
from lines import Line2D, LineSegment2D, has_crossing

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

def calculate_crossing_with(line, edges):
    crossings = 0
    for edge in edges:
        line_segment = edge_to_linesegment(edge)
        if has_crossing(line, line_segment):
            crossings += 1
    return crossings

def find_min_edge(selected_edges, lines, line_weights):
    weights = {}
    for edge in selected_edges:
        line_segment = edge_to_linesegment(edge)
        weights[edge] = 0.0
        for line in lines:
            if has_crossing(line, line_segment):
                weights[edge] += line_weights[line]
    #print weights
    #for edge in weights:
    #    print "%s => %s" % (edge, weights[edge])
    min_edge = min(weights, key=weights.get)
    (p,q) = min_edge
    if not p < q:
        min_edge = (q,p)
    #for edge in weights.keys():
    #    print "line = %s => weight = %s" % (edge, weights[edge])
    #print "returned min_edge= %s -> %s" % min_edge
    return min_edge


def compute_spanning_tree(points, lines):
    points = copy.deepcopy(points)
    lines = copy.deepcopy(lines)
    lines = preprocess_lines(lines)
    solution = []
    number_of_crossings = {}
    weights = {}
    graph = create_graph(points)
    while len(points) > 1:
        for line in lines:
            number_of_crossings[line] = calculate_crossing_with(line, solution)
            weights[line] = 2.**(number_of_crossings[line])
        #print "line weights = %s" % weights
        #print weights
        min_edge = find_min_edge(graph.get_edges(), lines, weights)
        (p,q) = min_edge
        #assert p < q
        graph.merge_cc_with_vertics(p,q)
        #format_string = "points = %s, point p=%s, q=%s, solution=%s" %\
        #        (points, p, q, solution)
        #print format_string
        if p in points:
            points.remove(p)
        elif q in points:
            points.remove(q)
        else:
            raise StandardError()
        solution.append(min_edge)
    print "final solution: %s" % solution
    return solution


if __name__ == '__main__':
    points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)]
    l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
    l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
    l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
    lines = [l1, l2, l3]
    solution = compute_spanning_tree(points, lines)
    import plotting
    plotting.plot(points, lines, solution)

