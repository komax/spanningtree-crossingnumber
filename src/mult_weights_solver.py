'''
computes a spanning tree for a point set s.t. it has low crossing number to
the line set, using the multiplicative weights method
'''

from datagenerator import Line2D, LineSegment2D

def preprocess_lines(lines):
    # TODO remove unnecessary lines
    return lines

def generate_edges(points):
    edges = [(p,q) for p in points
             for q in points
             if points.index(p) < points.index(q)]
    return tuplelist(edges)

def edge_to_linesegment(edge):
    p, q = edge
    return LineSegment2D(p, q)

def has_crossing(line, line_seg):
    if (line.slope == line_seg):
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

def connected_components(points, edges):
    # TODO compute connected components
    pass

def edges_crossing_connected_components(c_components):
    # TODO update implementation
    edges = []
    for connected_component in c_components:
        for other_c_component in c_components:
            if c_components.index(connected_components) <
            c_components.index(other_c_component):
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
        edges_between_c_components =
                edges_crossing_connected_components(c_components)
        min_edge = find_min_edge(edges_between_c_components, lines, weights)
        (p,q) = min_edge
        points.remove(p)
        edges.append(min_edge)
    return edges

