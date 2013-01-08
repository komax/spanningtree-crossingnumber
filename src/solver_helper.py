''' offers routines important to all different solvers '''

import math

def get_edges(points):
    ''' combine each point from the point set to the other, so that the
    resulting graph is fully connected. A edge is a 2-tuple
    '''
    edges = []
    for p in points:
        for q in points:
            if points.index(p) < points.index(q):
                edges.append((p,q))
    return edges

def euclidean_distance(p, q):
    ''' calculates the 2-D euclidean distance for two points '''
    (xp, yp) = p
    (xq, yq) = q
    return math.sqrt((xp+xq)**2 + (yp+yq)**2)

