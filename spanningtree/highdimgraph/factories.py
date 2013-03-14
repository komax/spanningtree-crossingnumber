'''
Created on Mar 8, 2013

@author: max
'''

from model import PointSet, Edges, HighDimGraph
from spanningtree import np
import math
import random

def create_uniform_points(n, d):
    point_range = n
    np_points = np.random.randint(0, point_range, size=(n,d))
    eps = 0.1
    eps_points = np.random.uniform(-eps, eps, size=(n,d))
    np_points = np_points + eps_points
    point_set = PointSet(np_points, n, d)
    point_set.name = 'uniform'
    return point_set

def create_grid_points(n, d):
    assert d == 2
    np_points = np.zeros(shape=(n,d),dtype=float)
    root_n = int(math.ceil(math.sqrt(n)))
    eps = 0.1
    x = 0.0
    y = 0.0
    row = 0
    diff = 50.0
    for i in range(root_n):
        y = 0.0
        for j in range(root_n):
            x_eps = random.uniform(-eps, eps)
            y_eps = random.uniform(-eps, eps)
            np_points[row] = np.array((x + x_eps, y + y_eps))
            row += 1
            y += diff
        x += diff
    assert row == n
    point_set = PointSet(np_points, n, d)
    point_set.name = 'grid'
    return point_set

def create_pointset(np_array, n, d, name):
    point_set = PointSet(np_array, n, d)
    point_set.name = name
    return point_set

def create_all_edges(n):
    adj_matrix = np.ones((n, n), dtype=bool)
    for i in range(0, n):
        adj_matrix[i, i] = False
    edges = Edges(n, adj_matrix)
    return edges

def create_solution_edges(n):
    sol_matrix = np.zeros((n, n), dtype=bool)
    solution = Edges(n, sol_matrix)
    return solution

def create_uniform_graph(n, d):
    points = create_uniform_points(n, d)
    edges = create_all_edges(n)
    return HighDimGraph(points, edges, n, d)

def create_grid_graph(n, d):
    points = create_grid_points(n, d)
    edges = create_all_edges(n)
    return HighDimGraph(points, edges, n, d)

def create_graph(points, n, d, name):
    point_set = create_pointset(points, n, d, name)
    edges = create_all_edges(n)
    return HighDimGraph(point_set, edges, n, d)
