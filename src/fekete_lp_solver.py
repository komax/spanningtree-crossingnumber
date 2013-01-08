'''
implements a solver that uses Fekete, Luebbecke LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
from lines import Line2D, LineSegment2D, has_crossing
from solver_helper import get_edges
from gurobipy import quicksum
import copy
import math
import itertools

def nonempty_subsets(points):
    '''
    returns all non empty subsets from the points set as generator
    '''
    for i in range(1,len(points)):
        for subset in itertools.combinations(points, i):
            yield list(subset)

def cut(subset, edges):
    '''
    returns as generator all edges ij containing i in subset and j not in
    subset (or the other way round)
    '''
    #cut_edges = []
    for i in subset:
        for (i,j) in edges.select(i, '*'):
            if j not in subset:
                yield (i,j)
                #cut_edges.append((i,j))
        for (j,i) in edges.select('*', i):
            if j not in subset:
                yield (j,i)
                #cut_edges.append((j,i))
    #return cut_edges

# global variables for decision variables
x = {}
t = 0

def create_lp(points, edges, lines):
    '''
    create a gurobi model containing LP formulation like that in the Fekete
    paper
    '''
    lambda_lp = grb.Model("fekete_lp_2d")
    n = len(points)
    for (p,q) in edges:
        x[p,q] = lambda_lp.addVar(# TODO maybe needed: obj=euclidean_distance(p,q),
                name='edge|%s - %s|' % (p,q))
    t = lambda_lp.addVar(obj=1.0, vtype=grb.GRB.INTEGER)

    lambda_lp.modelSense = grb.GRB.MINIMIZE

    lambda_lp.update()

    # correct number of edges
    lambda_lp.addConstr(quicksum(x[i,j] for (i,j) in edges) == (n-1))

    subsets = nonempty_subsets(points)
    # connectivity constraints
    for subset in subsets:
        lambda_lp.addConstr(quicksum(x[i,j] for (i,j) in cut(subset, edges))
                >= 1)

    # bound crossing number
    for line in lines:
        s = quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
            LineSegment2D(p,q)))
        if s != 0.0:
            lambda_lp.addConstr(s <= t)

    return lambda_lp


def solve_lp(lambda_lp):
    '''
    update (if needed) and solve the LP
    '''
    lambda_lp.update()
    lambda_lp.optimize()

    if lambda_lp.status == grb.GRB.status.OPTIMAL:
        return

def round_and_update_lp(edges, solution, alpha):
    '''
    find edges that are in the fractional solution, round them up and update the
    LP model

    returns the selected edges in the fractional solution (of this iteration)
    '''
    # TODO or find only the heaviest edge and bound it to 1
    (max_i, max_j) = (None, None)
    max_val = None
    for (i,j) in edges:
        if x[i,j].X > 1./3. and x[i,j].X > max_val:
            (max_i, max_j) = (i,j)
            max_val = x[i,j].X

    x[max_i,max_j].lb = 1.
    x[max_i,max_j].ub = 1.
    edges.remove((max_i,max_j))
    if max_i > max_j:
        (max_i, max_j) = (max_j, max_i)
    return [(max_i, max_j)]
    #round_edges = []
    #for (i,j) in edges:
    #    if (i,j) not in solution and x[i,j].X >= 1./alpha:
    #        x[i,j].lb = 1.
    #        x[i,j].ub = 1.
    #        if i > j:
    #            (i,j) = (j,i)
    #        round_edges.append((i,j))
    #return round_edges'''


def compute_spanning_tree(points, lines, alpha=2.0):
    solution = []
    n = len(points)
    edges = grb.tuplelist(get_edges(points))

    #i = 1
    number_of_edges = 0
    lp_model = create_lp(points, edges, lines)

    while number_of_edges < n-1:
        #print "round i=%s" % i
        solve_lp(lp_model)
        for var in lp_model.getVars():
            print var
        round_edges = round_and_update_lp(edges, solution, alpha)
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
        #i += 1
    return solution

def main():
    points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.),
            (4.25, 5.)]
    l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
    l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
    l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
    lines = [l1, l2, l3]
    solution = compute_spanning_tree(points, lines)
    print solution
    import plotting
    plotting.plot(points, lines, solution)

if __name__ == '__main__':
    main()
