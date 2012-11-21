"""
 module generates n 2D points in range 0..100.0
 and also 2D lines
"""

import random
import math

def generate_points_uniformly(n, ub=100.0):
    points = [(random.uniform(0,ub), random.uniform(0,ub))
            for i in range(n)]
    # TODO check for colinearity
    return points

def generate_points_grid(n):
    root_n = int(math.ceil(math.sqrt(n)))
    x = 0.0
    y = 0.0
    points = []
    for i in range(root_n):
        y = 0.0
        for j in range(root_n):
            points.append((x,y))
            y += 1.0
        x += 1.0
    assert len(points) == n
    return points

class Line2D:
    def __init__(self,p,q):
        assert p != q
        if p > q:
            (p, q) = (q, p)
        self.p = p
        self.q = q
        (x1, y1) = p
        (x2, y2) = q
        if x1 == x2:
            self.slope = 0.0
        else:
            self.slope = (y1 - y2) / (x1 - x2)
        self.y_intercept = y1 - self.slope * x1

    def __call__(self,x):
       y = self.slope * x + self.y_intercept
       return y

    def __str__(self):
       return 'Line2D(slope=%s, y_intercept=%s, points=%s)' % (self.slope,
               self.y_intercept,(self.p,self.q))

    def __repr__(self):
        return self.__str__()

    def is_on(self, p):
        (x,y) = p
        y_line = self(x)
        y_diff = math.fabs(y_line - y)
        return y_diff < 1e-13

class LineSegment2D(Line2D):
    def __init__(self, p, q):
        Line2D.__init__(self, p, q)

    def is_between(self, x):
        if self.slope > 0.0:
            return self.p[0] <= x <= self.q[0]
        else:
            return self.p[0] <= x <= self.q[0]

    def is_on(self, p):
        (x,y) = p
        if self.is_between(x):
            return Line2D.is_on(self, p)

    def __call__(self, x):
        if self.is_between(x):
            return Line2D.__call__(self, x)

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
