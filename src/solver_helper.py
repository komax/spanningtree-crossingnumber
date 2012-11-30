''' offers routines important to all different solvers '''

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

