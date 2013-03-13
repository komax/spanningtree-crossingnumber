'''
implements a solver that uses Fekete, Luebbecke LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import numpy as np
import gurobipy as grb
from spanningtree.highdimgraph import *
from spanningtree.helper.gurobi_helper import set_up_model
from gurobipy import quicksum
import math
import itertools

def nonempty_subsets(n):
    '''
    returns all non empty subsets from the points set as generator
    '''
    points = range(0,n)
    for i in range(1,n):
        for subset in itertools.combinations(points, i):
            yield list(subset)

def cut(subset, edges):
    '''
    returns as generator all edges ij containing i in subset and j not in
    subset (or the other way round)
    '''
    #cut_edges = []
    for (i,j) in edges:
        if i in subset and j not in subset:
            yield (i,j)
        elif i not in subset and j in subset:
            yield (i,j)
#    for i in subset:
#        for (p,q) in edges:
#            if j not in subset:
#                if i < j:
#                    yield (i,j)
#                else:
#                    yield (j,i)
#                #cut_edges.append((i,j))
#        for (j,i) in edges.select('*', i):
#            if j not in subset:
#                yield (j,i)
#                #cut_edges.append((j,i))
    #return cut_edges

# global variables for decision variables
x = {}
t = 0

def create_lp(graph):
    '''
    create a gurobi model containing LP formulation like that in the Fekete
    paper
    '''
    lambda_lp = set_up_model("fekete_lp_2d")
    #lambda_lp.setParam('Cuts', 3)
    number_of_edges = 0
    points = graph.point_set
    n = graph.n
    edges = graph.edges
    for (p,q) in edges:
        x[p,q] = lambda_lp.addVar(#obj=graph.euclidean_distance(p,q),
                name='edge|%s - %s|' % (p,q))
    t = lambda_lp.addVar(obj=1.0)#, vtype=grb.GRB.INTEGER)

    lambda_lp.modelSense = grb.GRB.MINIMIZE

    lambda_lp.update()

    # correct number of edges
    lambda_lp.addConstr(quicksum(x[i,j] for (i,j) in edges) == (n-1))

    subsets = nonempty_subsets(n)
    # connectivity constraints
    for subset in subsets:
        lambda_lp.addConstr(quicksum(x[i,j] for (i,j) in cut(subset, edges))
                >= 1)

    lines = graph.lines
    # bound crossing number
    for line in lines:
        s = quicksum(x[p,q] for (p,q) in edges if crossing.has_crossing(line,
            graph.get_line_segment(p,q)))
        if s != 0.0:
            lambda_lp.addConstr(s <= t)
    return lambda_lp


def solve_lp(lambda_lp, graph):
    '''
    update (if needed) and solve the LP
    '''
    lambda_lp.update()
    lambda_lp.optimize()

    if lambda_lp.status == grb.GRB.status.OPTIMAL:
        return
    else:
        format_string = 'Vars:\n'
        for var in lambda_lp.getVars():
            format_string += "%s\n" % var
        #for constr in lambda_lp.getConstrs():
        #    format_string += "%s\n" % constr
        print "number of lines = %s" % len(graph.lines)
        print '%s\nlp model=%s' % (format_string,lambda_lp)
        import spanningtree.plotting
        spanningtree.plotting.plot(graph)
        raise StandardError('Model infeasible')

def round_and_update_lp(graph, alpha):
    '''
    find edges that are in the fractional solution, round them up and update the
    LP model

    returns the selected edges in the fractional solution (of this iteration)
    '''
    edges = graph.edges
    solution = graph.solution
    (max_i, max_j) = (None, None)
    max_val = None
    for (i,j) in edges:
        if x[i,j].X > max_val and x[i,j] > 1./3.:
            (max_i, max_j) = (i,j)
            max_val = x[i,j].X

    x[max_i,max_j].lb = 1.
    x[max_i,max_j].ub = 1.
    edges.update(max_i,max_j, False)
    solution.update(max_i, max_j, True)
    return
#    if max_i > max_j:
#        (max_i, max_j) = (max_j, max_i)
#    return [(max_i, max_j)]
    #round_edges = []
    #for (i,j) in edges:
    #    if (i,j) not in solution and x[i,j].X >= 1./alpha:
    #        x[i,j].lb = 1.
    #        x[i,j].ub = 1.
    #        if i > j:
    #            (i,j) = (j,i)
    #        round_edges.append((i,j))
    #return round_edges'''


def compute_spanning_tree(graph, alpha=2.0):
    solution = graph.solution
    n = graph.n

    iterations = 0
    lp_model = create_lp(graph)

    while len(solution) < n-1:
        solve_lp(lp_model, graph)
        # printing all variables of LP
        #for var in lp_model.getVars():
        #    print var
        round_and_update_lp(graph, alpha)
        iterations += 1
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
