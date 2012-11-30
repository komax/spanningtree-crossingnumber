'''
implements a solver that uses Fekete, Luebbecke LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
from lines import Line2D, LineSegment2D, has_crossing
from gurobipy import quicksum
import copy
import math
import random

def solve_lp_and_round(points, lines, t):
    # TODO debug lp formulation
    pass
    '''gamma_lp = grb.Model("sariels_lp_2d")
    edges = get_edges(points)
    print "edges %s" % edges
    x = {}
    for (p,q) in edges:
        x[p,q] = gamma_lp.addVar(obj=euclidean_distance(p,q), name='edge|%s - %s|' % (p,q))

    gamma_lp.modelSense = grb.GRB.MINIMIZE

    gamma_lp.update()

    # crossing constraints
    for line in lines:
        s = quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
            LineSegment2D(p,q)))
        if s != 0.0:
                gamma_lp.addConstr(
                        #quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
                        #LineSegment2D(p,q))) <= t)
                        s <= t)

    # connectivity constraint
    for p in points:
        gamma_lp.addConstr(
                quicksum(x[p,q] for q in points if points.index(p) <
                    points.index(q)) +
                quicksum(x[q,p] for q in points if points.index(p) >
                    points.index(q))
                >= 1)

    gamma_lp.optimize()

    if gamma_lp.status == grb.GRB.status.OPTIMAL:
        round_solution = []
        for (p,q) in edges:
            print  x[p,q]
            if (12. * x[p,q].X) >= 1./12.:
                round_solution.append((p,q))
        return round_solution
        '''



def compute_spanning_tree(points, lines):
    # TODO implement this
    solution = []
    n = len(points)
    edges = get_edges(points)
    lines = preprocess_lines(lines)

    i = 1
    number_of_edges = 0

    while number_of_edges < n-1:
        print "round i=%s" % i
        lp_model = solve_lp(points, edges, lines)
        round_edges = round_and_update_lp(lp_model, edges)
        number_of_round_edges = len(round_edges)
        if  number_of_edges + number_of_round_edges <= n-1:
            # TODO check if resulting support graph is planar or has crossings
            solution += round_edges
            number_of_edges += number_of_round_edges
        else:
            # TODO check if resulting support graph is planar or has crossings
            l = (n-1) - number_of_edges
            solution += round_edges[:l]
            number_of_edges = n-1
            break
        i += 1
    return solution

def main():
    points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.),
            (4.25, 5.)]
    l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
    l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
    l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
    lines = [l1, l2, l3]
    solution = compute_spanning_tree(points, lines)
    import plotting
    plotting.plot(points, lines, solution)

if __name__ == '__main__':
    main()
