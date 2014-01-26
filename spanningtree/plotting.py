"""
plots all points and the lines in the plane. Using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from helper.numpy_helpers import partition


def plot(graph, verbose=False, save_to_file=None, show=True,
         plot_lines=False):
    assert save_to_file or show
    assert graph.d == 2
    '''
    plots points as blue circles, lines as red ones and the edges from the
    spanning tree as green line segments
    '''
    points = graph.point_set.points
    (xs, ys) = partition(points)
    x_range = np.arange(xs.min(), xs.max(), 0.25)
    y_min, y_max = ys.min(), ys.max()

    plt.figure()
    if plot_lines:
        # first plot lines
        l = list(graph.lines)
        assert l
        for line in graph.lines:
            line_x = []
            line_y = []
            for x in x_range:
                y = line.call([x])
                if y_min <= y <= y_max:
                    line_x.append(x)
                    line_y.append([y])
                plt.plot(line_x, line_y, 'r', zorder=1)
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
    plt.scatter(xs, ys, s=120, zorder=3)
    if save_to_file:
        filename = save_to_file + '.png'
        plt.savefig(filename)
    if show:
        plt.show()


def main():
    from highdimgraph.factories import create_grid_graph
    graph = create_grid_graph(2 ** 2, 2)
    # graph = create_uniform_graph(2, 2)
    #graph.create_all_lines()
    graph.create_random_lines()
    #graph.create_stabbing_lines()
    assert graph.lines
    graph.preprocess_lines()
    assert graph.lines
    import solvers.mult_weights_solver as mws

    mws.compute_spanning_tree(graph)
    plot(graph)

if __name__ == '__main__':
    main()
