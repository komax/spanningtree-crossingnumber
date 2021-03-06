#! /usr/bin/env python
'''
unit test (suite) module checks functionality of all
other programmed modules
'''

import unittest
import spanningtree.solvers.mult_weights_solver as mwsolv
from spanningtree.highdimgraph.lines import HighDimLine, HighDimLineSegment
from spanningtree.highdimgraph.crossing import has_crossing
from spanningtree.helper.numpy_helpers import np_assert_allclose
from spanningtree.highdimgraph.factories import create_graph
import spanningtree.highdimgraph as highdimgraph
import spanningtree.solvers.hp_lp_solver as hplpsolv
import spanningtree.solvers.fekete_lp_solver as flpsolv
import spanningtree.solvers.opt_solver as opt_solver
import numpy as np


class CrossingTestCase(unittest.TestCase):
    def setUp(self):
        highdimgraph.crossing.new_crossing_registry()

    def test_has_crossing(self):
        line = HighDimLine(np.array([(3., 2), (5., 0)]))  # y = -x + 5
        line_segment = HighDimLineSegment(
                np.array([(3, 0.5), (5, 1.5)]))  # y = 0.5 x - 1
        # line_points[0] = segpoints[0] = intersection point
        line_points = np.array([(4., 1.), (5., 0.)])
        self.assertTrue(line.is_on(line_points[0]))
        self.assertTrue(line.is_on(line_points[1]))

        seg_points = np.copy(line_points)
        seg_points[1, 1] = 1.5
        self.assertTrue(line_segment.is_on(seg_points[0]))
        self.assertTrue(line_segment.is_on(seg_points[1]))

        line_val = line.call(np.array([5.]))
        np_assert_allclose(0.0, line_val)
        np_assert_allclose(line_points[..., -1], line(line_points[..., :-1]))
        self.assertTrue(has_crossing(line, line_segment))

    def test_has_no_crossing_linesegment_too_short(self):
        line = HighDimLine(np.array([[3., 2], [5, 0]]))  # y = -x + 5
        line_segment = HighDimLineSegment(
                np.array([[0., -1], [3, 0.5]]))  # y = 0.5 x - 1
        self.assertFalse(has_crossing(line, line_segment))

    def test_has_no_crossing_line_and_segment_parallel(self):
        line = HighDimLine(np.array([(3, 2), (5, 0)]))  # y = -x + 5
        line_segment = HighDimLineSegment(
                np.array([(0, 4), (6, -2)]))  # y = -x + 4
        self.assertFalse(has_crossing(line, line_segment))

    def test_has_also_crossing(self):
        line = HighDimLine(np.array([(3., 5.5), (5., 6.5)]))  # y = 0.5x + 4
        line_segment = HighDimLineSegment(
                np.array([(5., 7.), (6., 4.)]))  # y = -3x + 22
        self.assertTrue(has_crossing(line, line_segment))


class PreprocessingLinesTestCase(unittest.TestCase):
    def test_is_above_works(self):
        line = HighDimLine(np.array(((3, 2), (5, 0))))  # y = -x + 5
        p = np.array((1, 4.5))
        self.assertTrue(line.is_above(p))
        self.assertFalse(line.is_below(p))

    def test_is_below_works(self):
        line = HighDimLine(np.array(((3, 0.5), (5, 1.5))))  # y = 0.5 x - 1
        p = np.array((4., -1))
        self.assertTrue(line.is_below(p))
        self.assertFalse(line.is_above(p))

    def test_is_on_works(self):
        line = HighDimLine(np.array(((3, 0.5), (5, 1.5))))  # y = 0.5 x - 1
        p = np.array((4, 1))
        self.assertTrue(line.is_on(p))
        self.assertFalse(line.is_below(p))
        self.assertFalse(line.is_above(p))

    def test_preprocessing_lines_omits_duplicates(self):
        points = np.array([(2., 2.), (6., 4.), (3., 6.), (5., 7.), (4.25, 5.)])
        graph = create_graph(points, 5, 2, "foo")
        l1 = HighDimLine(np.array([(2., 6.), (3., 2.)]))  # y = -4x + 14
        l2 = HighDimLine(np.array([(2., 3.), (6., 5.)]))  # y = 0.5x + 2
        l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)]))  # y = 0.5x + 4
        # duplicates part
        l4 = HighDimLine(np.array(((2.5, 6.), (3.5, 2.))))  # y = -4x + 16
        l5 = HighDimLine(np.array(((2., 2.5), (6., 4.5))))  # y = 0.5x + 1.5
        l6 = HighDimLine(np.array(((3., 5.), (5., 6.))))  # y = 0.5x + 3.5
        # lines outside of point set
        # above
        l7 = HighDimLine(np.array([(0., 7.), (1., 10.)]))  # y = 3 x + 7
        # below
        l8 = HighDimLine(np.array([(0., -1.), (6., 1.)]))  # y = 1/3 x - 1
        lines = [l1, l2, l3, l4, l5, l6, l7, l8]
        graph.lines = lines
        graph.preprocess_lines()
        result = graph.lines
        self.assertEqual(len(result), 3)
        self.assertFalse(l7 in result)
        self.assertFalse(l8 in result)
        self.assertTrue(l1 in result or l4 in result)
        self.assertTrue(l2 in result or l5 in result)
        self.assertTrue(l3 in result or l6 in result)


class LinePreprocessingConnectedComponentsTestCase(unittest.TestCase):
    def setUp(self):
        # setting up example graph
        points = np.array([(8.5, 2), (10.5, 3.5), (4., 4.), (5., 5.), (8.5,
            5.5), (9., 7.), (10.5, 6.5)])
        graph = create_graph(points, 7, 2, "example")
        # first connected component
        graph.solution.update(0, 1, True)
        graph.edges.update(0, 1, False)
        # second connected component
        graph.solution.update(2, 3, True)
        graph.edges.update(2, 3, False)
        # third connected component
        graph.solution.update(4, 5, True)
        graph.solution.update(5, 6, True)
        graph.edges.update(4, 5, False)
        graph.edges.update(5, 6, False)
        graph.edges.update(4, 6, False)
        # update information in graph about connected components
        graph.compute_connected_components()
        self.graph = graph

    def test_preprocess_on_connected_components(self):
        # setting up lines
        l1 = HighDimLine(np.array([(4., 5.), (5., 4.)]))
        l2 = HighDimLine(np.array([(8.5, 3.5), (9.5, 2.)]))
        l3 = HighDimLine(np.array([(8.5, 5.5), (10., 6.5)]))
        lines = [l1, l2, l3]
        self.graph.lines = lines
        # TODO implement this method
        self.graph.preprocess_lines_on_ccs()
        expected_lines = self.graph.lines
        self.assertEqual(len(expected_lines), 2)
        self.assertTrue(l2 in expected_lines)
        self.assertTrue(l3 in expected_lines)


class DFSTestCase(unittest.TestCase):
    def setUp(self):
        point_set = np.array([[0], [1], [2], [3], [4]])
        self.graph = create_graph(point_set, 5, 1, "foo")

    def test_dfs(self):
        self.graph.solution.update(0, 1, True)
        self.graph.solution.update(1, 2, True)
        self.graph.solution.update(2, 3, True)
        self.graph.solution.update(1, 4, True)
        expectation = [0, 1, 2, 3, 4]
        expectation2 = [0, 1, 4, 2, 3]
        result = list(self.graph.dfs(0))
        self.assertTrue(expectation == result or expectation2 == result)

    def test_cycling(self):
        self.graph.solution.update(0, 1, True)
        self.graph.solution.update(1, 2, True)
        self.graph.solution.update(2, 3, True)
        self.graph.solution.update(0, 3, True)
        self.assertTrue(self.graph.has_cycle())

    def test_no_cycling(self):
        self.graph.solution.update(0, 1, True)
        self.graph.solution.update(1, 2, True)
        self.graph.solution.update(2, 3, True)
        self.graph.solution.update(1, 4, True)
        self.assertFalse(self.graph.has_cycle())

    def test_only_singletons(self):
        self.assertFalse(self.graph.is_spanning_tree())

    def test_is_spanningtree(self):
        self.graph.solution.update(0, 1, True)
        self.graph.solution.update(1, 2, True)
        self.graph.solution.update(2, 3, True)
        self.graph.solution.update(1, 4, True)
        self.assertTrue(self.graph.is_spanning_tree())

    def test_cycling_no_spanningtree(self):
        self.graph.solution.update(0, 1, True)
        self.graph.solution.update(1, 2, True)
        self.graph.solution.update(2, 3, True)
        self.graph.solution.update(0, 3, True)
        self.assertFalse(self.graph.is_spanning_tree())


class GraphTestCase(unittest.TestCase):
    def setUp(self):
        point_set = np.array([[0], [1], [2], [3]])
        self.graph = create_graph(point_set, 4, 1, "foo")

    def test_connected_components(self):
        cc1 = self.graph.connected_components.get_connected_component(1)
        cc2 = self.graph.connected_components.get_connected_component(2)
        self.assertFalse(cc1 == cc2)
        cc3 = self.graph.connected_components.get_connected_component(1)
        self.assertTrue(cc1 == cc3)

    def test_adjacent(self):
        self.assertTrue(self.graph.edges.has_edge(1, 2))
        self.assertFalse(self.graph.edges.has_edge(1, 1))
        adjacent_vertices = list(self.graph.edges.adj_nodes(1))
        adjacent_vertices.sort()
        self.assertEqual(adjacent_vertices, [0, 2, 3])

    def test_get_edges(self):
        edges = list(self.graph.edges)
        expected_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.assertEqual(edges, expected_edges)

    def test_merge_connected_components(self):
        self.graph.compute_connected_components()
        self.graph.merge_cc(0, 1)
        self.graph.compute_spanning_tree_on_ccs()
        edges = list(self.graph.edges)
        expected_edges = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.assertEqual(edges, expected_edges)

    def test_merge_cc_with_vertices(self):
        self.graph.connected_components.merge_by_vertices(0, 1)
        self.graph.connected_components.merge_by_vertices(0, 2)
        self.graph.connected_components.merge_by_vertices(3, 1)
        self.graph.compute_spanning_tree_on_ccs()
        edges = list(self.graph.solution)
        self.assertEqual(edges, [])

    def test_get_connected_component(self):
        self.graph.connected_components.merge_by_vertices(0, 1)
        connected_component =\
            list(self.graph.connected_components.get_connected_component(1))
        connected_component.sort()
        self.assertEqual(connected_component, [0, 1])

    def test_cannot_find_connected_component(self):
        self.assertRaises(StandardError,
                self.graph.connected_components.get_connected_component, 5)


class PlanarityTransformationTestCase(unittest.TestCase):
    def setUp(self):
        points = np.array([(3., 3.), (4.5, 2.), (8.5, 3.5), (6.5, 6.5)])
        graph = create_graph(points, 4, 2, "example")
        graph.solution.update(0, 3, True)
        # self crossing edges between (0,2) and (1,3)
        graph.solution.update(0, 2, True)
        graph.solution.update(1, 3, True)
        self.graph = graph

    def test_planarity(self):
        self.assertTrue(self.graph.is_spanning_tree())
        iterations = self.graph.make_planar()
        self.assertEquals(iterations, 2)
        #for (i, j) in self.graph.solution:
        #    print (i,j)
        self.assertTrue(self.graph.is_spanning_tree())
        solution = list(self.graph.solution)
        self.assertTrue((0, 1) in solution)
        self.assertTrue((2, 3) in solution)
        self.assertTrue((0, 3) in solution)


class MultWeightsSolvingTestCase(unittest.TestCase):
    def setUp(self):
        points = np.array([(2., 2.), (6., 4.), (3., 6.), (5., 7.), (4.25, 5.)])
        self.graph = create_graph(points, 5, 2, "foo")

        l1 = HighDimLine(np.array([(2., 6.), (3., 2.)]))  # y = -4x + 14
        l2 = HighDimLine(np.array([(2., 3.), (6., 5.)]))  # y = 0.5x + 2
        l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)]))  # y = 0.5x + 4
        self.graph.lines = [l1, l2, l3]

    def test_solution(self):
        mwsolv.compute_spanning_tree(self.graph)
        solution = list(self.graph.solution)
        self.assertEqual(len(solution), 4)
        self.assertTrue((2, 3) in solution)
        self.assertTrue((0, 1) in solution)
        self.assertTrue((2, 4) in solution or
                (3, 4) in solution)
        self.assertTrue((1, 4) in solution)
        self.assertTrue(self.graph.is_spanning_tree())


class ConnectedComponentsTestCase(unittest.TestCase):
    def test_compute_spanning_tree_on_ccs(self):
        points = np.array([(0., 3.), (3., 4.), (9., 10.), (7., 8.)])
        graph = create_graph(points, 4, 2, "foo")
        edges = [(1, 0),
                  (0, 2),
                  (2, 3),
                  (1, 2)]
        for (i, j) in edges:
            graph.solution.update(i, j, True)
        graph.compute_connected_components()
        ccs = graph.connected_components
        self.assertEqual(len(ccs), 1)
        cc = ccs.get_connected_component(2)
        points_indices = range(0, 4)
        self.assertItemsEqual(cc, points_indices)

        graph.compute_spanning_tree_on_ccs()
        sol_edges = [(0, 2), (0, 1), (2, 3)]
        self.assertItemsEqual(sol_edges, graph.solution)

    def test_results_one_connected_component(self):
        points = np.array([(0., 3.), (3., 4.), (9., 10.), (7., 8.),
            (5., 6.), (2., 1.)])
        graph = create_graph(points, 6, 2, "foo")
        edges = [(1, 0),
                  (5, 1),
                  (1, 4),
                  (3, 4),
                  (2, 3)]
        for (i, j) in edges:
            graph.solution.update(i, j, True)
        graph.compute_connected_components()
        ccs = graph.connected_components
        self.assertEqual(len(ccs), 1)
        cc = ccs.get_connected_component(3)
        points_indices = range(0, 6)
        self.assertItemsEqual(cc, points_indices)

        graph.compute_spanning_tree_on_ccs()
        cc_edges = graph.solution
        for (i, j) in edges:
            if i < j:
                edge = (i, j)
            else:
                edge = (j, i)
            self.assertTrue(edge in cc_edges,
                    "%s not in %s" % (edge, cc_edges))

    def test_results_two_connected_components(self):
        points = np.array([(0., 3.), (3., 4.), (9., 10.), (7., 8.),
            (5., 6.), (2., 1.)])
        graph = create_graph(points, 6, 2, "foo")
        edges = [(1, 0),
                  (5, 1),
                  # (1, 4), now two connected components
                  (3, 4),
                  (2, 3)]
        for (i, j) in edges:
            graph.solution.update(i, j, True)
        graph.compute_connected_components()
        ccs = graph.connected_components
        self.assertEqual(len(ccs), 2)
        c1 = set([0, 1, 5])
        self.assertItemsEqual(c1, ccs.get_connected_component(0))
        c1_edges = [(0, 1), (1, 5)]
        c2 = set([2, 3, 4])
        self.assertItemsEqual(c2, ccs.get_connected_component(3))
        self.assertNotEqual(ccs.get_connected_component(0),
                ccs.get_connected_component(3))
        c2_edges = [(3, 4), (2, 3)]
        expected_sol = c1_edges + c2_edges
        self.assertItemsEqual(expected_sol, graph.solution)


class SarielsLPSolvingTestCase(unittest.TestCase):
    def setUp(self):
        points = np.array([(2., 2.), (6., 4.), (3., 6.), (5., 7.), (4.25, 5.)])
        self.graph = create_graph(points, 5, 2, "foo")

        l1 = HighDimLine(np.array([(2., 6.), (3., 2.)]))  # y = -4x + 14
        l2 = HighDimLine(np.array([(2., 3.), (6., 5.)]))  # y = 0.5x + 2
        l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)]))  # y = 0.5x + 4
        self.graph.lines = [l1, l2, l3]

    def test_solution(self):
        hplpsolv.compute_spanning_tree(self.graph)
        solution = list(self.graph.solution)
        self.assertEqual(len(solution), 4)
        self.assertTrue((2, 3) in solution)
        self.assertTrue((0, 1) in solution)
        self.assertTrue((2, 4) in solution or
                (3, 4) in solution)
        self.assertTrue((1, 4) in solution)
        self.assertTrue(self.graph.is_spanning_tree())


class FeketeLPSolvingTestCase(unittest.TestCase):
    def setUp(self):
        points = np.array([(2., 2.), (6., 4.), (3., 6.), (5., 7.), (4.25, 5.)])
        self.graph = create_graph(points, 5, 2, "foo")

        l1 = HighDimLine(np.array([(2., 6.), (3., 2.)]))  # y = -4x + 14
        l2 = HighDimLine(np.array([(2., 3.), (6., 5.)]))  # y = 0.5x + 2
        l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)]))  # y = 0.5x + 4
        self.graph.lines = [l1, l2, l3]

    def test_solution(self):
        flpsolv.compute_spanning_tree(self.graph)
        solution = list(self.graph.solution)
        self.assertEqual(len(solution), 4)
        self.assertTrue((2, 3) in solution)
        self.assertTrue((0, 1) in solution)
        self.assertTrue((2, 4) in solution or
                (3, 4) in solution)
        self.assertTrue((1, 4) in solution)
        self.assertTrue(self.graph.is_spanning_tree())

    def test_subsets(self):
        points_size = 4
        subsets_gen = flpsolv.nonempty_subsets(points_size)
        subsets = list(subsets_gen)
        for subset in subsets:
            print subset
        self.assertEqual(len(subsets), 2 ** points_size - 2)


class OptSolverTestCase(unittest.TestCase):
    def setUp(self):
        points = np.array([(2., 2.), (6., 4.), (3., 6.), (5., 7.), (4.25, 5.)])
        self.graph = create_graph(points, 5, 2, "foo")

        l1 = HighDimLine(np.array([(2., 6.), (3., 2.)]))  # y = -4x + 14
        l2 = HighDimLine(np.array([(2., 3.), (6., 5.)]))  # y = 0.5x + 2
        l3 = HighDimLine(np.array([(3., 5.5), (5., 6.5)]))  # y = 0.5x + 4
        self.graph.lines = [l1, l2, l3]

    def test_solution(self):
        opt_solver.compute_spanning_tree(self.graph)
        solution = list(self.graph.solution)
        self.assertEqual(len(solution), 4)
        self.assertTrue((2, 3) in solution)
        self.assertTrue((0, 1) in solution)
        self.assertTrue((2, 4) in solution or
                (3, 4) in solution)
        self.assertTrue((1, 4) in solution)
        self.assertTrue(self.graph.is_spanning_tree())

if __name__ == '__main__':
    unittest.main()
