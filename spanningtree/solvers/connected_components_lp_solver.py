'''
Created on Mar 18, 2013

@author: max
'''
from mult_weights_solver import add_edge_to_solution_merge_ccs

def solve_lp(graph, points):
    lines = graph.lines
    gamma_lp = set_up_model("connected_components_lp_2d")
    graph.compute_connected_components()
    connected_components = graph.connected_components
    len_ccs = len(connected_components)
    edges = list(graph.edges.iter_subset(points))
    x = {}
    for (p,q) in edges:
        x[p,q] = gamma_lp.addVar(#obj=graph.euclidean_distance(p,q),
                                 name='edge|%s - %s|' % (p,q))

    t = gamma_lp.addVar(obj=1.0)
    gamma_lp.modelSense = grb.GRB.MINIMIZE

    gamma_lp.update()
    
    # correct number of edges
    gamma_lp.addConstr(quicksum(x[i,j] for (i,j) in edges) == (len_ccs-1))

    # crossing number range:
    gamma_lp.addConstr(0 <= t <= math.sqrt(len_ccs))

    # crossing constraints
    for line in lines:
        s = quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
            graph.get_line_segment(p,q)))
        if s != 0.0:
                gamma_lp.addConstr(
                        #quicksum(x[p,q] for (p,q) in edges if has_crossing(line,
                        #LineSegment2D(p,q))) <= t)
                        s <= t)

    # connectivity constraint
    for cc1 in connected_components:
        gamma_lp.addConstr(
                quicksum(x[p,q] for q in points if p < q and
                         p in cc1 and q not in cc1) +
                quicksum(x[q,p] for q in points if p > q and
                         p in cc1 and q not in cc1)
                >= 1)

    gamma_lp.optimize()

    if gamma_lp.status == grb.GRB.status.OPTIMAL:
        return
    else:
        format_string = 'Vars:\n'
        for var in gamma_lp.getVars():
            format_string += "%s\n" % var
        print "number of lines = %s" % len(graph.lines)
        print '%s\nlp model=%s' % (format_string,gamma_lp)
        import spanningtree.plotting
        spanningtree.plotting.plot(graph)
        raise StandardError('Model infeasible')
    
def add_best_edge(graph):
    edges = graph.edges
    ccs = graph.connected_components
    solution = graph.solution
    (max_i, max_j) = (None, None)
    max_val = None
    for (i,j) in edges:
        cc_i = ccs.get_connected_component(i)
        cc_j = ccs.get_connected_component(j) 
        if cc_i != cc_j and x[i,j].X > max_val and x[i,j] > 1./3.:
            (max_i, max_j) = (i,j)
            max_val = x[i,j].X

    x[max_i,max_j].lb = 1.
    x[max_i,max_j].ub = 1.
    edges.update(max_i,max_j, False)
    solution.update(max_i, max_j, True)
    return

def compute_spanning_tree(graph):
    n = graph.n
    stored_lines = graph.lines[:]
    remaining_points = range(0,n)
    solution = graph.solution
    lines = graph.lines
    iterations = 1
    while len(remaining_points) > 1:
        t = estimate_t(remaining_points)
        round_edges = solve_lp_and_round(graph, remaining_points)
        graph.compute_connected_components()
        connected_components = graph.connected_components
        if not has_proper_no_of_connected_components(connected_components, remaining_points):
            put_back_round_edges(graph, round_edges)
            continue
        new_point_set = []
        for connected_component in connected_components:
            p = random.sample(connected_component, 1)[0]
            new_point_set.append(p)
        remaining_points = new_point_set
        lines = graph.preprocess_lines(remaining_points)
        iterations += 1
    assert len(remaining_points) == 1
    graph.lines = stored_lines
    graph.compute_spanning_tree_on_ccs()
    return iterations

def main():
    # minimal example to find optimal spanning tree
    points = np.array([(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)])
    graph = create_graph(points, 5, 2)
    l1 = HighDimLine(np.array([(2., 6.), (3., 2.)])) # y = -4x + 14
    l2 = HighDimLine(np.array([(2., 3.), (6., 5.)])) # y = 0.5x + 2
    l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)])) # y = 0.5x + 4
    lines = [l1, l2, l3]
    graph.lines = lines
    graph.preprocess_lines()
    compute_spanning_tree(graph)
    print "crossing number = %s" % graph.crossing_number()
    import plotting
    plotting.plot(graph)

if __name__ == '__main__':
    main()