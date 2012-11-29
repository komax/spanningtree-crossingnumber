'''
implements a solver that uses Sariel Har-Paled LP formulation currently in
planar 2D
using Gurobi as standard solver
'''

import gurobipy as grb
from lines import Line2D, LineSegment2D, has_crossing
from gurobipy import quicksum
import copy
import math
import random

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
            if x[p,q].X >= 1./12.:
                round_solution.append((p,q))
        return round_solution

def has_proper_no_of_connected_components(points, ccs):
    # TODO in planar case always true. implement it
    no_connected_components = len(ccs)
    ratio_points = 19./20. * len(points)
    print "# connected components=%s <= %s, val=%s" % (no_connected_components,
            ratio_points, no_connected_components <= ratio_points)
    if no_connected_components <= ratio_points:
        return True
    else:
        return False

def connected_components(points, edges):
    remaining_points = copy.deepcopy(points)
    edges = grb.tuplelist(edges)
    ccs = []

    while remaining_points:
        p = remaining_points.pop()
        p_edges = edges.select(p, "*") + edges.select("*", p)
        queue = []

        for (u,v) in p_edges:
            if v == p:
                (u,v) = (v,u)
            if v in remaining_points:
                remaining_points.remove(v)
            queue.append(v)

        new_connected_component = [p]
        while queue:
            q = queue.pop(0)
            new_connected_component.append(q)
            q_edges = edges.select(q, "*") + edges.select("*", q)
            for (u,v) in q_edges:
                if v == q:
                    (u,v) = (v,u)
                if not v in new_connected_component and v in remaining_points:
                    queue.append(v)
                    remaining_points.remove(v)
        ccs.append(new_connected_component)
    return ccs

def estimate_t(points):
    return math.sqrt(len(points))

def compute_spanning_tree(points, lines):
    points = copy.deepcopy(points)
    lines = copy.deepcopy(lines)
    lines = preprocess_lines(lines)

    solution = []
    i = 1
    while len(points) > 1:
        points.sort()
        print "round %i" % i
        t = estimate_t(points)
        print "estimated t=%s" % t
        round_edges = solve_lp_and_round(points, lines, t)
        print "round edges %s" % round_edges
        ccs = connected_components(points, round_edges)
        if not has_proper_no_of_connected_components(points,
                ccs):
            continue
        new_point_set = []
        print "# of connected components %i" % len(ccs)
        for connected_component in ccs:
            assert len(connected_component) >= 1
            print "connected component |%s|" % connected_component
            repr_index = random.randint(0, len(connected_component)-1)
            p = connected_component[repr_index]
            new_point_set.append(p)
        points = new_point_set
        solution += round_edges
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
