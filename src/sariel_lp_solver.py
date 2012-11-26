'''
implements a solver that uses Sariel Har-Paled LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
from datagenerator import Line2D, LineSegment2D
import copy
import math

def preprocess_lines(lines):
    # TODO remove unnecessary lines
    return lines

def get_edges(points):
    edges = []
    for p in points:
        for q in points:
            if points.index(p) < points.index(q):
                edges.append((p,q))
    return edges

def euclidean_distance(p, q):
    (xp, yp) = p
    (xq, yq) = q
    return math.sqrt((xp+xq)**2 + (yp+yq)**2)


def solve_lp_and_round(points, lines, t):
    # TODO debug lp formulation
    gamma_lp = grb.Model("sariels_lp_2d")
    edges = get_edges(points)
    x = {}
    for (p,q) in edges:
        x[p,q] = gamma_lp.addVar(obj=euclidean_distance(p,q), name='edge|%s - %s|' % (p,q))

    gamma_lp.modelSense = grb.GRB.MINIMIZE

    gamma_lp.update()

    # crossing constraints
    for line in lines:
        gamma_lp.addConstr(
                quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
                    LineSegment2D(p,q))) <= t)

    # connectivity constraint
    for p in points:
        gamma_lp.addConstr(
                quicksum(x[p,q] for q in points if p != q) >= 1)

    gamma_lp.optimize()

    if gamma_lp.status == grb.GRB.status.OPTIMAL:
        round_solution = []
        for (p,q) in edges:
            if x[p,q].X >= 1./12.:
                round_solution.append((p,q))
        return round_solution

def has_proper_no_of_connected_components(points, connected_components):
    # TODO in planar case always true. implement it
    #if len(connected_components) >= (19./20. * len(points)):
    #    return False
    return True

def connected_components(points, edges):
    # TODO implementation, unit test this implementation
    edges = grb.tuplelist(edges)
    connected_components = []
    for p in points:
        p_edges = edges.select(p, "*") + edges.select("*", p)
        for connected_component in connected_components:
            if p in connected_component:
                for (u,v) in p_edges:
                    if v == p:
                        (u,v) = (v,u)
                    if v not in connected_component:
                        connected_component.append(v)
        else:
            # this is a new connected component
            new_connected_component = [p]
            for (u,v) in p_edges:
                if v == p:
                    (u,v) = (v,u)
                if v not in new_connected_component:
                    new_connected_component.append(v)
            connected_components.append(new_connected_component)
    return connected_components

def compute_spanning_tree(points, lines, t):
    points = copy.deepcopy(points)
    lines = copy.deepcopy(lines)
    lines = preprocess_lines(lines)

    solution = []

    while len(points) > 1:
        round_edges = solve_lp_and_round(points, lines, t)
        if not has_proper_no_of_connected_components(points, round_edges):
            continue
        new_point_set = []
        for c in connected_components(points, round_edges):
            assert len(c) >= 1
            p = c[0]
            new_point_set.append(p)
        points = new_point_set
        solution += round_edges
    return solution

