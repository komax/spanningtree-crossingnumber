"""
plots all points and the lines in the plane. Using matplotlib
"""

import matplotlib.pyplot as plt
import math
from pylab import frange
from highdimgraph import create_uniform_graph, create_grid_graph, partition

def plot(graph):
    assert graph.d == 2
    # FIXME update this implementation to numpy graphs
    '''
    plots points as blue circles, lines as red ones and the edges from the
    spanning tree as green line segments
    '''
    points = graph.point_set.points
    (xs, ys) = partition(points)
#    for (x,y) in points:
#        xs.append(x)
#        ys.append(y)
#    x_frange = frange(max(xs))

    # first plot lines
    for line in graph.lines:
        plt.plot(xs, line(xs), 'r', zorder=1)
    # then plot solution
    xlines = []
    ylines = []
    
    
    for (i, j) in graph.solution:
        (x1, y1) = partition(points[i])
        (x2, y2) = partition(points[j])
        xlines.append(x1)
        xlines.append(x2)
        xlines.append(None)
        # same for ys
        ylines.append(y1)
        ylines.append(y2)
        ylines.append(None)
    plt.plot(xlines, ylines, 'g', zorder=2)
    # then plot points
    plt.scatter(xs, ys,  s=120, zorder=3)
    plt.show()

def main():
    graph = create_grid_graph(2**2, 2)
    #graph = create_uniform_graph(2, 2)
    graph.create_all_lines()
    graph.preprocess_lines()
    #graph.create_stabbing_lines()
    #points = dtgen.generate_points_uniformly(6, 100.0)
    #lines = dtgen.generate_lines(points)
    import mult_weights_solver as mws
    
    mws.compute_spanning_tree(graph)
    plot(graph)

if __name__ == '__main__':
    main()

