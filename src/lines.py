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

    def is_above(self, p):
        (x,y) = p
        y_line = self(x)
        return y > y_line and not self.is_on(p)

    def is_below(self, p):
        (x,y) = p
        y_line = self(x)
        return y < y_line and not self.is_on(p)

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
        else:
            return False

    def __call__(self, x):
        if self.is_between(x):
            return Line2D.__call__(self, x)

def has_crossing(line, line_seg):
    '''
    Has line a crossing with the line segment
    '''
    if line.slope == line_seg.slope:
        return False
    else:
        x_s = (line.y_intercept - line_seg.y_intercept) / (line_seg.slope -
                line.slope)
        y_s = line(x_s)
        return line_seg.is_between(x_s)

def calculate_crossing_with(line, edges):
    '''
    for a given line calculate the crossing number (int) over all edges
    '''
    crossings = 0
    for (p,q) in edges:
        line_segment = LineSegment2D(p,q)
        if has_crossing(line, line_segment):
            crossings += 1
    return crossings

def calculate_crossings(lines, solution):
    '''
    for all lines and edges in the solution compute the overall crossings
    '''
    crossing_number = 0
    for line in lines:
        crossing_number += calculate_crossing_with(line, solution)
    return crossing_number

def maximum_crossing_number(lines, solution):
    '''
    for all lines and edges in the solution compute the maximum crossing number
    '''
    max_crossing_number = 0
    for line in lines:
        crossing_number = calculate_crossing_with(line, solution)
        if crossing_number > max_crossing_number:
            max_crossing_number = crossing_number
    return max_crossing_number

crossing_number = maximum_crossing_number

def minimum_crossing_number(lines, solution):
    '''
    for all lines and edges in the solution compute the minimum crossing number
    '''
    min_crossing_number = len(solution)
    for line in lines:
        crossing_number = calculate_crossing_with(line, solution)
        if crossing_number < min_crossing_number:
            min_crossing_number = crossing_number
    return min_crossing_number

def partition_points(line, points):
    ''' partitioning of point set with discriminative function line
        (points above the line as tuples , points below the line as tuples)
        both parts can be empty
    '''
    above_points = []
    below_points = []
    for p in points:
        if line.is_on(p):
            return ()
        elif line.is_above(p):
            above_points.append(p)
        elif line.is_below(p):
            below_points.append(p)
        else:
            raise StandardError('can not find point p=%s on line=%s' %
                    (p,line))
    return (tuple(above_points), tuple(below_points))

def preprocess_lines(lines, points):
    ''' removes lines that have same partitioning of the point set as
        equivalent ones
        lines above or below the point set are also removed
    '''
    lines_dict = {}
    for line in lines:
        partition_tuple = partition_points(line, points)
        if not partition_tuple:
            # skip this line, because one point is on this line
            continue
        elif not partition_tuple[0] or not partition_tuple[1]:
            # above or below part is empty, skip line
            continue
        elif lines_dict.has_key(partition_tuple):
            # skip this line, there is one equivalent line stored
            continue
        else:
            # new equivalent class, store this line
            lines_dict[partition_tuple] = line
    return lines_dict.values()

