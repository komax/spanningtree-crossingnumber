'''
implements a solver that uses Sariel Har-Paled LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
from spanningtree.highdimgraph import *
from spanningtree.helper.gurobi_helper import set_up_model
from spanningtree.highdimgraph.crossing import has_crossing
from gurobipy import quicksum
import math
import random

def solve_lp_and_round(graph, points):
    '''
    creates a Gurobi model for Sariel LP formulation and round it
    deterministically

    return the selection of edges that are in the fractional solution
    '''
    lines = graph.lines
    gamma_lp = set_up_model("sariels_lp_2d")
    n = len(points)
    edges = list(graph.edges.iter_subset(points))
    x = {}
    for (p,q) in edges:
        x[p,q] = gamma_lp.addVar(obj=graph.euclidean_distance(p,q), name='edge|%s - %s|' % (p,q))

    t = gamma_lp.addVar(obj=1.0)
    gamma_lp.modelSense = grb.GRB.MINIMIZE

    gamma_lp.update()

    # crossing number range:
    gamma_lp.addConstr(0 <= t <= math.sqrt(n))

    # crossing constraints
    for line in lines:
        s = quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
            graph.get_line_segment(p,q)))
        if s != 0.0:
                gamma_lp.addConstr(
                        #quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
                        #LineSegment2D(p,q))) <= t)
                        s <= t)

    # connectivity constraint
    for p in points:
        gamma_lp.addConstr(
                quicksum(x[p,q] for q in points if p < q) +
                quicksum(x[q,p] for q in points if p > q)
                >= 1)

    gamma_lp.optimize()

    if gamma_lp.status == grb.GRB.status.OPTIMAL:
        round_solution = []
        for (p,q) in edges:
            val = x[p,q].X
            sample = random.random()
            if sample <= val:
            #if val >= 1./12.:
                graph.solution.update(p, q, True)
                round_solution.append((p,q))
        return round_solution

def has_proper_no_of_connected_components(connected_components, points):
    '''
    boolean function that checks if the number of connected components is below
    19/20 * number of points
    '''
    no_connected_components = len(connected_components)
    ratio_points = 19./20. * len(points)
    if no_connected_components <= ratio_points:
        return True
    else:
        return False

def put_back_round_edges(graph, round_edges):
    for (i, j) in round_edges:
        graph.solution.update(i, j, False)
    return

def estimate_t(points):
    return math.sqrt(len(points))

def compute_spanning_tree(graph):
    n = graph.n
    stored_lines = graph.lines[:]
    remaining_points = range(0,n)
    solution = graph.solution
    lines = graph.lines
    iterations = 1
    while len(remaining_points) > 1:
        t = estimate_t(remaining_points)
        round_edges = solve_lp_and_round(graph, remaining_points)
        graph.compute_connected_components()
        connected_components = graph.connected_components
        if not has_proper_no_of_connected_components(connected_components, remaining_points):
            put_back_round_edges(graph, round_edges)
            continue
        new_point_set = []
        for connected_component in connected_components:
            p = random.sample(connected_component, 1)[0]
            new_point_set.append(p)
        remaining_points = new_point_set
        lines = graph.preprocess_lines(remaining_points)
        iterations += 1
    assert len(remaining_points) == 1
    graph.lines = stored_lines
    graph.compute_spanning_tree_on_ccs()
    return iterations

def main():
    # minimal example to find optimal spanning tree
    points = np.array([(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)])
    graph = create_graph(points, 5, 2)
    l1 = HighDimLine(np.array([(2., 6.), (3., 2.)])) # y = -4x + 14
    l2 = HighDimLine(np.array([(2., 3.), (6., 5.)])) # y = 0.5x + 2
    l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)])) # y = 0.5x + 4
    lines = [l1, l2, l3]
    graph.lines = lines
    graph.preprocess_lines()
    compute_spanning_tree(graph)
    print "crossing number = %s" % graph.crossing_number()
    import plotting
    plotting.plot(graph)

if __name__ == '__main__':
    main()
