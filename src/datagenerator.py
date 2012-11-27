"""
 module generates n 2D points in range 0..100.0
 and also 2D lines
"""

import random
import math
from lines import Line2D, LineSegment2D

def generate_points_uniformly(n, ub=100.0):
    points = [(random.uniform(0,ub), random.uniform(0,ub))
            for i in range(n)]
    # TODO check for colinearity
    return points

def generate_points_grid(n):
    root_n = int(math.ceil(math.sqrt(n)))
    eps = 0.1
    x = 0.0
    y = 0.0
    points = []
    for i in range(root_n):
        y = 0.0
        for j in range(root_n):
            x_eps = random.uniform(-eps,eps)
            y_eps = random.uniform(-eps,eps)
            points.append((x+x_eps,y+y_eps))
            y += 5.0
        x += 5.0
    assert len(points) == n
    return points

def create_lines(p,q, eps):
    (x1, y1) = p
    (x2, y2) = q
    lines = []
    y_delta = math.fabs(y1 - y2)
    delta = y_delta * eps
    if x1 == x2:
        # special case if point are in a grid
        x1l = x1 - delta
        x1r = x1 + delta
        x2l = x2 - delta
        x2r = x2 + delta
        lines.append(Line2D((x1l,y1),(x2l,y2)))
        lines.append(Line2D((x1r,y1),(x2r,y2)))
        lines.append(Line2D((x1l,y1),(x2r,y2)))
        lines.append(Line2D((x1r,y1),(x2l,y2)))
    else:
        y1u = y1 + delta
        y1b = y1 - delta
        y2u = y2 + delta
        y2b = y2 - delta
        lines.append(Line2D((x1,y1u),(x2,y2u)))
        lines.append(Line2D((x1,y1b),(x2,y2b)))
        lines.append(Line2D((x1,y1u),(x2,y2b)))
        lines.append(Line2D((x1,y1b),(x2,y2u)))
    return lines

def generate_lines(points, eps=0.1):
    lines = {}
    print points
    for p in points:
        for q in points:
            if points.index(p) < points.index(q):
                if not lines.has_key((p,q)):
                    # create all different lines
                    pq_lines = create_lines(p,q, eps)
                    lines[p,q] = pq_lines
    for (p,q) in lines.keys():
        print "%s -> %s" % (p,q)
    line_set = []
    for pq_lines in lines.values():
        line_set = line_set + pq_lines
    return line_set

def main():
    points = generate_points_uniformly(4)
    print points
    lines = generate_lines(points)
    print lines
    print "main finished work"

if __name__ == '__main__':
    main()
