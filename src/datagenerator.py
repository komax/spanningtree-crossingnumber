"""
 module generates n 2D points in range 0..100.0
 and also 2D lines
"""

import random
import math
from lines import clean_lines, get_line

def generate_points_uniformly(n, lb=0.0, ub=100.0):
    '''
    sample n 2-D points in range lb..ub
    '''
    if ub <= n:
        ub *= n
    points = [(random.uniform(lb,ub), random.uniform(lb,ub))
            for i in range(n)]
    return points

def generate_points_grid(n):
    '''
    sample a grid with n points. All points are slightly pertubated to permit
    vertical and horizontal lines
    '''
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
    '''
    for two points p,q compute all four possible separation lines
    '''
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
        lines.append(get_line((x1l,y1),(x2l,y2)))
        lines.append(get_line((x1r,y1),(x2r,y2)))
        lines.append(get_line((x1l,y1),(x2r,y2)))
        lines.append(get_line((x1r,y1),(x2l,y2)))
    else:
        y1u = y1 + delta
        y1b = y1 - delta
        y2u = y2 + delta
        y2b = y2 - delta
        lines.append(get_line((x1,y1u),(x2,y2u)))
        lines.append(get_line((x1,y1b),(x2,y2b)))
        lines.append(get_line((x1,y1u),(x2,y2b)))
        lines.append(get_line((x1,y1b),(x2,y2u)))
    return lines

def generate_lines(points, eps=0.1):
    ''' compute all possible seperators (lines) on the point set. There maybe
        duplicates within this set
    '''
    clean_lines()
    lines = {}
    for p in points:
        for q in points:
            if points.index(p) < points.index(q):
                if not lines.has_key((p,q)):
                    # create all different lines
                    pq_lines = create_lines(p,q, eps)
                    lines[p,q] = pq_lines
    line_set = []
    for pq_lines in lines.values():
        line_set = line_set + pq_lines
    return line_set

def generate_random_lines(n, points):
    '''
    generate n random lines within the value range of the point set
    '''
    assert n > 0
    clean_lines()
    min_x = +100000.
    max_x = -100000.
    min_y = min_x
    max_y = max_x
    for (x,y) in points:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > min_y:
            max_y = y
    if max_x > max_y:
        ub = max_x
    else:
        ub = max_y
    if min_x > min_y:
        lb = min_y
    else:
        lb = min_x
    p_points = generate_points_uniformly(n,lb, ub)
    q_points = generate_points_uniformly(n,lb, ub)
    i = 0
    line_set = []
    for p in p_points:
        q = q_points[i]
        line_set.append(get_line(p,q))
        i += 1
    return line_set

def main():
    '''
    for minor testing
    '''
    points = generate_points_uniformly(4)
    print points
    lines = generate_lines(points)
    print lines
    print "main finished work"

if __name__ == '__main__':
    main()
