'''
implements a solver that uses Fekete, Luebbecke LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
import numpy as np
import spanningtree.highdimgraph.crossing as crossing
from spanningtree.helper.gurobi_helper import set_up_model
from gurobipy import quicksum
import itertools


def nonempty_subsets(n):
    '''
    returns all non empty subsets from the points set as generator
    '''
    required_subsets = 2 ** (n - 1) - 1
    number_of_subsets = 0
    points = range(n)

    subset_length = 1
    while number_of_subsets <= required_subsets:
        for subset in itertools.combinations(points, subset_length):
            number_of_subsets += 1
            yield subset
        subset_length += 1


def cut(subset, edges):
    '''
    returns as generator all edges ij containing i in subset and j not in
    subset (or the other way round)
    '''
    #cut_edges = []
    for (i, j) in edges:
        if i in subset and j not in subset:
            yield (i, j)
        elif i not in subset and j in subset:
            yield (i, j)

def cut_edges(n, edges):
    for subset in nonempty_subsets(n):
        yield [(i,j) for (i,j) in cut(subset, edges)]

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
    n = graph.n
    edges = graph.edges
    for (p, q) in edges:
        x[p, q] = lambda_lp.addVar(name='edge|%s - %s|' % (p, q))
    t = lambda_lp.addVar(obj=1.0)

    lambda_lp.modelSense = grb.GRB.MINIMIZE

    lambda_lp.update()

    # correct number of edges
    lambda_lp.addConstr(quicksum(x[i, j] for (i, j) in edges) == (n - 1))

#    subsets = nonempty_subsets(n)
#    # connectivity constraints
#    for subset in subsets:
#        lambda_lp.addConstr(quicksum(x[i,j] for (i,j) in cut(subset, edges))
#                >= 1)
    # connectivity constraint
    connected_components = graph.connected_components
    for cc1 in connected_components:
        lambda_lp.addConstr(quicksum(x[p, q] for (p, q) in edges if p < q and
                         p in cc1 and q not in cc1) +
                quicksum(x[q, p] for (q, p) in edges if p > q and
                         p in cc1 and q not in cc1)
                >= 1.)

    lines = graph.lines
    # bound crossing number
    for line in lines:
        s = quicksum(x[p, q] for (p, q) in edges if crossing.has_crossing(line,
            graph.get_line_segment(p, q)))
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
        print '%s\nlp model=%s' % (format_string, lambda_lp)
        import spanningtree.plotting
        spanningtree.plotting.plot(graph)
        raise StandardError('Model infeasible')


def round_and_update_lp(graph, alpha):
    '''
    find edges that are in the fractional solution, round them up
    and update the LP model

    returns the number of trails that failed to add a heavy weight edge
    '''
    edges = graph.edges
    ccs = graph.connected_components
    solution = graph.solution
    (max_i, max_j) = (None, None)
    max_val = None
    # counter for fails: selecting a heavy weight edge within one cc
    trails = 0
    for (i, j) in edges:
        cc_i = ccs.get_connected_component(i)
        cc_j = ccs.get_connected_component(j)
        if cc_i == cc_j:
            trails += 1
        if cc_i != cc_j and x[i, j].X > max_val:
            (max_i, max_j) = (i, j)
            max_val = x[i, j].X

    x[max_i, max_j].lb = 1.
    x[max_i, max_j].ub = 1.
    edges.update(max_i, max_j, False)
    solution.update(max_i, max_j, True)
    return trails


def compute_spanning_tree(graph, alpha=2.0):
    solution = graph.solution
    n = graph.n

    iterations = 0
    lp_model = create_lp(graph)
    
    # failed trails of adding a heavy weight edge
    trails = list()

    while len(solution) < n - 1:
        graph.compute_connected_components()
        solve_lp(lp_model, graph)
        # printing all variables of LP
        #for var in lp_model.getVars():
        #    print var
        trail_i = round_and_update_lp(graph, alpha)
        trails.append(trail_i)
        iterations += 1
    meaned_trail = np.mean(trails)
    return (iterations, meaned_trail)


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
