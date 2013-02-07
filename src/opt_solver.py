'''
computes the optimal solution with the Fekete, Luebbecke LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
from lines import get_line, get_line_segment, has_crossing
from solver_helper import get_edges
from gurobipy import quicksum
import copy
import math
import itertools

def nonempty_subsets(points):
    '''
    same as in fekete
    '''
    for i in range(1,len(points)):
        for subset in itertools.combinations(points, i):
            yield list(subset)

def cut(subset, edges):
    '''
    same as in fekete
    '''
    for i in subset:
        for (i,j) in edges.select(i, '*'):
            if j not in subset:
                yield (i,j)
        for (j,i) in edges.select('*', i):
            if j not in subset:
                yield (j,i)

x = {}
t = 0

def create_ip(points, edges, lines):
    '''
    creates a gurobi model containing Fekete IP formulation
    '''
    lambda_ip = grb.Model("fekete_ip_2d")
    n = len(points)
    for (p,q) in edges:
        x[p,q] = lambda_ip.addVar(# TODO maybe needed: obj=euclidean_distance(p,q),
                vtype=grb.GRB.BINARY,name='edge|%s - %s|' % (p,q))
    t = lambda_ip.addVar(obj=1.0, vtype=grb.GRB.INTEGER)

    lambda_ip.modelSense = grb.GRB.MINIMIZE
    lambda_ip.setParam('Cuts', 3)
    lambda_ip.setParam('Threads', 4)

    lambda_ip.update()

    # correct number of edges
    lambda_ip.addConstr(quicksum(x[i,j] for (i,j) in edges) == (n-1))

    # connectivity constraints
    for subset in nonempty_subsets(points):
        lambda_ip.addConstr(quicksum(x[i,j] for (i,j) in cut(subset, edges))
                >= 1)

    # bound crossing number
    for line in lines:
        s = quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
            get_line_segment(p,q)))
        if s != 0.0:
            lambda_ip.addConstr(s <= t)

    return lambda_ip


def solve_ip(lambda_ip):
    '''
    computes solution in the IP
    '''
    # TODO is this update call necessary
    lambda_ip.update()
    lambda_ip.optimize()

    if lambda_ip.status == grb.GRB.status.OPTIMAL:
        return

def create_solution(edges):
    '''
    select from decision variable all edges in the solution
    '''
    solution = []
    for (i,j) in edges:
        if x[i,j].X == 1.:
            if i > j:
                (i,j) = (j,i)
            solution.append((i,j))
    return solution


def compute_spanning_tree(points, lines):
    solution = []
    n = len(points)
    edges = grb.tuplelist(get_edges(points))

    ip_model = create_ip(points, edges, lines)
    solve_ip(ip_model)
    solution = create_solution(edges)
    return solution

def main():
    points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.),
            (4.25, 5.)]
    l1 = get_line((2., 6.), (3., 2.)) # y = -4x + 14
    l2 = get_line((2., 3.), (6., 5.)) # y = 0.5x + 2
    l3 = get_line((3., 5.5), (5., 6.5)) # y = 0.5x + 4
    lines = [l1, l2, l3]
    solution = compute_spanning_tree(points, lines)
    print solution
    import plotting
    plotting.plot(points, lines, solution)

if __name__ == '__main__':
    main()
