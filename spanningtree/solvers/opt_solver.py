'''
computes the optimal solution with the Fekete, Luebbecke LP formulation
currently in planar 2D using Gurobi as standard solver
'''

import gurobipy as grb
import spanningtree.highdimgraph.crossing as crossing
from spanningtree.helper.gurobi_helper import set_up_model
#from fekete_lp_solver import nonempty_subsets, cut
from fekete_lp_solver import cut_edges
from gurobipy import quicksum

x = {}
t = 0


def create_ip(graph):
    '''
    creates a gurobi model containing Fekete IP formulation
    '''
    lambda_ip = set_up_model("fekete_ip_2d")
    #lambda_ip.setParam('Cuts', 3)
    n = graph.n
    edges = graph.edges
    for (p, q) in edges:
        x[p, q] = lambda_ip.addVar(vtype=grb.GRB.BINARY,
                name='edge|%s - %s|' % (p, q))
    t = lambda_ip.addVar(obj=1.0, vtype=grb.GRB.INTEGER)

    lambda_ip.modelSense = grb.GRB.MINIMIZE

    lambda_ip.update()

    # correct number of edges
    lambda_ip.addConstr(quicksum(x[i, j] for (i, j) in edges) == (n - 1))

#    # connectivity constraints
#    connected_components = graph.connected_components
#    for cc1 in connected_components:
#        lambda_ip.addConstr(quicksum(x[p,q] for (p,q) in edges if p < q and\
#                         p in cc1 and q not in cc1) +
#                quicksum(x[q,p] for (q,p) in edges if p > q and\
#                         p in cc1 and q not in cc1)
#                >= 1.)
    #for subset in nonempty_subsets(n):
    #    lambda_ip.addConstr(quicksum(x[i, j] for (i, j) in cut(subset, edges))
    #            >= 1)
    global subset_edges
    subset_edges = cut_edges(n, edges)
#    for edges in subset_edges:
#        lambda_ip.addConstr(quicksum(x[i, j] for (i, j) in edges)
#                >= 1)
    global is_solution
    is_solution = check_ip_solution(graph)

    lines = graph.lines
    # bound crossing number
    for line in lines:
        s = quicksum(x[p, q] for (p, q) in edges if crossing.has_crossing(line,
            graph.get_line_segment(p, q)))
        if s != 0.0:
            lambda_ip.addConstr(s <= t)
    return lambda_ip

subset_edges = None
is_solution = None

def mycallback(model, where):
    global is_solution
    if where == grb.GRB.callback.MIPSOL:
        #status =  model.get(grb.GRB.STATUS)
        #print "current status=%s, is optimal=%s" % (status, status ==
        #        grb.GRB.status.OPTIMAL)
        print "number of solutions=%s" %\
            model.cbGet(grb.GRB.callback.MIPSOL_SOLCNT)

        print model.cbGetSolution(model.getVars())

        if not is_solution(model):# and\
          #model.cbGet(grb.GRB.callback.MIPSOL_SOLCNT) <= 0:
            print "no spanning tree"
            has_next = True
            while has_next:
                try:
                    edges = subset_edges.next()
                    edge_sum = quicksum(x[i,j] for (i,j) in edges)
                    print "added lazy constraint"
                    model.cbLazy(edge_sum >= 1)
                except StopIteration:
                    print "added all constraints"
                    has_next = False
        else:
            print "model is feasible; found a spanningtree"
    else:
        pass

def solve_ip(lambda_ip):
    '''
    computes solution in the IP
    '''
    lambda_ip.params.DualReductions = 0
    lambda_ip.optimize(mycallback)
    #lambda_ip.optimize()

    #lambda_ip.printStats()
    #print lambda_ip.status
    if lambda_ip.status == grb.GRB.status.OPTIMAL:
        return

def check_ip_solution(graph):
    edges = graph.edges
    def is_spanning_tree(model):
        print "checking spanning_tree"
        for (i,j) in edges:
            val = model.cbGetSolution(x[i,j])
            print val
            if val == 1:
                graph.solution.update(i, j, True)
        result = graph.is_spanning_tree()
        print "is_spanningtree=%s" % result
        for (i,j) in edges:
            val = model.cbGetSolution(x[i,j])
            if val == 1:
                graph.solution.update(i, j, False)
        return result
    return is_spanning_tree

def create_solution(graph):
    '''
    select from decision variable all edges in the solution
    '''
    edges = graph.edges
    solution = graph.solution
    for (i, j) in edges:
        print x[i,j]
        if x[i, j].X == 1.:
            solution.update(i, j, True)
            edges.update(i, j, False)
    return


def compute_spanning_tree(graph):
    ip_model = create_ip(graph)
    solve_ip(ip_model)
    create_solution(graph)
    #print "is_spanning_tree = %s" % is_solution(ip_model)
    #assert is_solution()
    print 'is a spanningtree = %s' % graph.is_spanning_tree()
    return 1


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
