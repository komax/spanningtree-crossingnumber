"""
plots all points and the lines in the plane. Using matplotlib
"""

import matplotlib.pyplot as plt
import math
from pylab import frange
import datagenerator as dtgen

def plot(points, lines, solution):
    '''
    plots points as blue circles, lines as red ones and the edges from the
    spanning tree as green line segments
    '''
    xs = []
    ys = []
    for (x,y) in points:
        xs.append(x)
        ys.append(y)
    x_frange = frange(max(xs))

    # first plot lines
    for line in lines:
        plt.plot(x_frange, line(x_frange), 'r', zorder=1)
    # then plot solution
    xlines = []
    ylines = []
    for ((x1, y1), (x2, y2)) in solution:
        xlines.append(x1)
        xlines.append(x2)
        xlines.append(None)
        # same for ys
        ylines.append(y1)
        ylines.append(y2)
        ylines.append(None)
    plt.plot(xlines, ylines, 'g', zorder=2)
    # then plot points
    plt.scatter(xs, ys, s=120, zorder=3)
    plt.show()

def main():
    points = dtgen.generate_points_grid(5**2)
    #points = dtgen.generate_points_uniformly(6, 100.0)
    lines = dtgen.generate_lines(points)
    import mult_weights_solver as mws
    solution = mws.compute_spanning_tree(points, lines)
    plot(points, lines, solution)

if __name__ == '__main__':
    main()

