'''
Created on Mar 8, 2013

@author: max
'''

def create_uniform_points(n, d):
    point_set = PointSet(n, d)
    point_set.name = 'uniform'
    return point_set

def create_grid_points(n, d):
    assert d == 2
    point_set = PointSet(n, d)
    root_n = int(math.ceil(math.sqrt(n)))
    eps = 0.1
    x = 0.0
    y = 0.0
    row = 0
    for i in range(root_n):
        y = 0.0
        for j in range(root_n):
            x_eps = random.uniform(-eps, eps)
            y_eps = random.uniform(-eps, eps)
            point_set.points[row] = np.array((x + x_eps, y + y_eps))
            row += 1
            y += 5.0
        x += 5.0
    assert row == n
    point_set.name = 'grid'
    return point_set

def create_pointset(np_array, n, d, name):
    assert np_array.shape == (n, d)
    point_set = PointSet(n, d)
    point_set.points = np_array
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