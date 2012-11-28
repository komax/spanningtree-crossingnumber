'''
Simple implmentation of lines and line segments in 2D. Offers also a function
to check if a line and segment crosses
'''

import math

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

    def __key(self):
        return (self.slope, self.y_intercept)

    def __eq__(a, b):
        return a.__key() == b.__key()

    def __hash__(self):
        return hash(self.__key())

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
    
    def __key(self):
        return (self.slope, self.y_intercept)

    def __eq__(a, b):
        return a.__key() == b.__key()

    def __hash__(self):
        return hash(self.__key())

    def is_between(self, x):
        return self.p[0] <= x <= self.q[0]

    def is_on(self, p):
        (x,y) = p
        if self.is_between(x):
            return Line2D.is_on(self, p)

    def __call__(self, x):
        if self.is_between(x):
            return Line2D.__call__(self, x)

def has_crossing(line, line_seg):
    if line.slope == line_seg.slope:
        return False
    else:
        x_s = (line.y_intercept - line_seg.y_intercept) / (line_seg.slope -
                line.slope)
        y_s = line(x_s)
        return line_seg.is_between(x_s)

