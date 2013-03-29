'''
Created on Mar 18, 2013

@author: max
'''

from spanningtree.helper.gurobi_helper import set_up_model
import gurobipy as grb
from gurobipy import quicksum
import math
from spanningtree.highdimgraph.crossing import has_crossing


def solve_lp(graph):
    points = range(graph.n)
    lines = graph.lines
    gamma_lp = set_up_model("connected_components_lp_2d")
    connected_components = graph.connected_components
    len_ccs = len(connected_components)
    edges = list(graph.edges.iter_subset(points))
    x = {}
    for (p, q) in edges:
        x[p, q] = gamma_lp.addVar(name='edge|%s - %s|' % (p, q))

    t = gamma_lp.addVar(obj=1.0)
    gamma_lp.modelSense = grb.GRB.MINIMIZE

    gamma_lp.update()

    # correct number of edges
    gamma_lp.addConstr(quicksum(x[i, j] for (i, j) in edges) == (len_ccs - 1))

    # crossing number range:
    gamma_lp.addConstr(0 <= t <= math.sqrt(len_ccs))

    # crossing constraints
    for line in lines:
        s = quicksum(x[p, q] for (p, q) in edges if has_crossing(line,
            graph.get_line_segment(p, q)))
        if s != 0.0:
                gamma_lp.addConstr(
                        s <= t)

    # connectivity constraint
    for cc1 in connected_components:
        gamma_lp.addConstr(quicksum(x[p, q] for (p, q) in edges if p < q and
                         p in cc1 and q not in cc1) +
                quicksum(x[q, p] for (q, p) in edges if p > q and
                         p in cc1 and q not in cc1)
                >= 1.)
    gamma_lp.optimize()

    if gamma_lp.status == grb.GRB.status.OPTIMAL:
        #print gamma_lp.getVars()
        #print x[2,3].X
        add_best_edge(graph, x)
        return
    else:
        format_string = 'Vars:\n'
        for var in gamma_lp.getVars():
            format_string += "%s\n" % var
        print "number of lines = %s" % len(graph.lines)
        print '%s\nlp model=%s' % (format_string, gamma_lp)
        import spanningtree.plotting
        spanningtree.plotting.plot(graph)
        raise StandardError('Model infeasible')


def add_best_edge(graph, x):
    edges = graph.edges
    ccs = graph.connected_components
    solution = graph.solution
    (max_i, max_j) = (None, None)
    max_val = None
    for (i, j) in edges:
        cc_i = ccs.get_connected_component(i)
        cc_j = ccs.get_connected_component(j)
        if cc_i != cc_j and x[i, j].X > max_val:
            (max_i, max_j) = (i, j)
            max_val = x[i, j].X

    graph.merge_cc(max_i, max_j)
    solution.update(max_i, max_j, True)
    return


def compute_spanning_tree(graph):
    #n = graph.n
    #stored_lines = graph.lines[:]
    #remaining_points = range(0,n)
    #lines = graph.lines
    iterations = 0
    while len(graph.connected_components) > 1:
        solve_lp(graph)
        #print x[2,3].X
        #add_best_edge(graph, x)
        #graph.compute_connected_components()
        #connected_components = graph.connected_components
        # TODO preprocess line set;
        # include only lines between connected components
        #lines = graph.preprocess_lines(remaining_points)
        iterations += 1
    #graph.lines = stored_lines
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
    print "crossing number = %s" % graph.crossing_number()
    import spanningtree.plotting as plotting
    plotting.plot(graph)

if __name__ == '__main__':
    main()
