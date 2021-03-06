'''
computes a spanning tree for a point set s.t. it has low crossing number to
the line set, using the multiplicative weights method
'''
from spanningtree.highdimgraph import crossing


def add_edge_to_solution_merge_ccs(graph, i, j):
    ccs = graph.connected_components
    cci = ccs.get_connected_component(i)
    ccj = ccs.get_connected_component(j)
    for p in cci:
        for q in ccj:
            graph.edges.update(p, q, False)
    ccs.merge(cci, ccj)
    graph.solution.update(i, j, True)


def find_min_edge(graph, line_weights):
    '''
    search for the minimum weight edge between two connected components with
    lowest crossing weight
    '''
    weights = {}
    for edge in graph.edges:
        (p, q) = edge
        line_segment = graph.get_line_segment(p, q)
        weights[edge] = 0.0
        for line in graph.lines:
            if crossing.has_crossing(line, line_segment):
                weights[edge] += line_weights[line]
    min_edge = min(weights, key=weights.get)
    (p, q) = min_edge
    assert p < q
    return min_edge


def compute_spanning_tree(graph):
    lines = graph.lines

    number_of_crossings = {}
    weights = {}
    iterations = 0

    while len(graph.connected_components) > 1:
        for line in lines:
            number_of_crossings[line] = graph.calculate_crossing_with(line)
            weights[line] = 2. ** (number_of_crossings[line])
        (i, j) = find_min_edge(graph, weights)
        add_edge_to_solution_merge_ccs(graph, i, j)
        iterations += 1
    return iterations


def main():
    # minimal example to find optimal spanning tree
    import numpy as np
    points = np.array([(2., 2.), (6., 4.), (3., 6.), (5., 7.), (4.25, 5.)])
    import spanningtree.highdimgraph.factories as factories
    graph = factories.create_graph(points, 5, 2, 'custom')
    from spanningtree.highdimgraph.lines import HighDimLine
    # y = -4x + 14
    l1 = HighDimLine(np.array([(2., 6.), (3., 2.)]))
    # y = 0.5x + 2
    l2 = HighDimLine(np.array([(2., 3.), (6., 5.)]))
    # y = 0.5x + 4
    l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)]))
    lines = [l1, l2, l3]
    graph.lines = lines
    graph.preprocess_lines()
    compute_spanning_tree(graph)
    assert graph.is_spanning_tree()
    print "crossing number = %s" % graph.crossing_number()
    import spanningtree.plotting as plotting
    plotting.plot(graph)

if __name__ == '__main__':
    main()
