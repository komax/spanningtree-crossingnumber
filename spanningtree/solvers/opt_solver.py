'''
computes the optimal solution with the Fekete, Luebbecke LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
import spanningtree.highdimgraph.crossing as crossing
from spanningtree.helper.gurobi_helper import set_up_model
from fekete_lp_solver import nonempty_subsets, cut
from gurobipy import quicksum
import math
import itertools

x = {}
t = 0

def create_ip(graph):
    '''
    creates a gurobi model containing Fekete IP formulation
    '''
    lambda_ip = set_up_model("fekete_ip_2d")
    lambda_ip.setParam('Cuts', 3)
    n = graph.n
    edges = graph.edges
    for (p,q) in edges:
        x[p,q] = lambda_ip.addVar(#obj=graph.euclidean_distance(p,q),
                vtype=grb.GRB.BINARY,name='edge|%s - %s|' % (p,q))
    t = lambda_ip.addVar(obj=1.0, vtype=grb.GRB.INTEGER)

    lambda_ip.modelSense = grb.GRB.MINIMIZE

    lambda_ip.update()

    # correct number of edges
    lambda_ip.addConstr(quicksum(x[i,j] for (i,j) in edges) == (n-1))

    # connectivity constraints
    for subset in nonempty_subsets(n):
        lambda_ip.addConstr(quicksum(x[i,j] for (i,j) in cut(subset, edges))
                >= 1)

    lines = graph.lines
    # bound crossing number
    for line in lines:
        s = quicksum(x[p,q] for (p,q) in edges if crossing.has_crossing(line,
            graph.get_line_segment(p,q)))
        if s != 0.0:
            lambda_ip.addConstr(s <= t)
    return lambda_ip


def solve_ip(lambda_ip):
    '''
    computes solution in the IP
    '''
    lambda_ip.optimize()

    if lambda_ip.status == grb.GRB.status.OPTIMAL:
        return

def create_solution(graph):
    '''
    select from decision variable all edges in the solution
    '''
    edges = graph.edges
    solution = graph.solution
    for (i,j) in edges:
        if x[i,j].X == 1.:
            solution.update(i,j, True)
            edges.update(i,j, False)
    return


def compute_spanning_tree(graph):
    ip_model = create_ip(graph)
    solve_ip(ip_model)
    create_solution(graph)
    return 1

def main():
    # minimal example to find optimal spanning tree
    import numpy as np
    points = np.array([(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)])
    import spanningtree.highdimgraph.factories as factories
    graph = factories.create_graph(points, 5, 2, 'custom')
    from spanningtree.highdimgraph.lines import HighDimLine
    l1 = HighDimLine(np.array([(2., 6.), (3., 2.)])) # y = -4x + 14
    l2 = HighDimLine(np.array([(2., 3.), (6., 5.)])) # y = 0.5x + 2
    l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)])) # y = 0.5x + 4
    lines = [l1, l2, l3]
    graph.lines = lines
    graph.preprocess_lines()
    compute_spanning_tree(graph)
    print "crossing number = %s" % graph.crossing_number()
    import spanningtree.plotting as plotting
    plotting.plot(graph)

if __name__ == '__main__':
    main()
