#! /usr/bin/env python
''' unit test (suite) module checks functionality of all other programmed modules '''

import unittest
import mult_weights_solver as mwsolv
from highdimgraph import HighDimLine, HighDimLineSegment, has_crossing
import highdimgraph
import sariel_lp_solver as slpsolv
import fekete_lp_solver as flpsolv
import opt_solver
import math
import numpy as np

class CrossingTestCase(unittest.TestCase):
    def test_has_crossing(self):
        line = HighDimLine(np.array([(3.,2), (5.,0)])) # y = -x + 5
        print line
        line_segment = HighDimLineSegment(np.array([(3,0.5), (5,1.5)])) # y = 0.5 x - 1
        print line_segment
        points = np.array([(4., 1.), (5., 0.)])
        np.testing.assert_almost_equal(points[..., -1], line(points[..., :-1]))
        np.testing.assert_almost_equal(points[..., -1], line_segment(points[..., :-1]))
        self.assertTrue(has_crossing(line, line_segment))

    def test_has_no_crossing_linesegment_too_short(self):
        line = HighDimLine(np.array([[3.,2], [5,0]])) # y = -x + 5
        line_segment = HighDimLineSegment(np.array([[0.,-1], [3,0.5]])) # y = 0.5 x - 1
        self.assertFalse(has_crossing(line, line_segment))

    def test_has_no_crossing_line_and_segment_parallel(self):
        line = HighDimLine(np.array([(3,2), (5,0)])) # y = -x + 5
        line_segment = HighDimLineSegment(np.array([(0, 4), (6, -2)])) # y = -x + 4
        self.assertFalse(has_crossing(line, line_segment))

    def test_has_also_crossing(self):
        line = HighDimLine(np.array([(3., 5.5), (5., 6.5)])) # y = 0.5x + 4
        line_segment = HighDimLineSegment(np.array([(5.,7.), (6., 4.)])) # y = -3 + 22
        self.assertTrue(has_crossing(line, line_segment))

#class PreprocessingLinesTestCase(unittest.TestCase):
#    def test_is_above_works(self):
#        line = Line2D((3,2), (5,0)) # y = -x + 5
#        p = (1,4.5)
#        self.assertTrue(line.is_above(p))
#        self.assertFalse(line.is_below(p))
#
#    def test_is_below_works(self):
#        line = Line2D((3,0.5), (5,1.5)) # y = 0.5 x - 1
#        p = (4, -1)
#        self.assertTrue(line.is_below(p))
#        self.assertFalse(line.is_above(p))
#
#    def test_is_on_works(self):
#        line = Line2D((3,0.5), (5,1.5)) # y = 0.5 x - 1
#        p = (4, 1)
#        self.assertTrue(line.is_on(p))
#        self.assertFalse(line.is_below(p))
#        self.assertFalse(line.is_above(p))
#
#    def test_preprocessing_lines_omits_duplicates(self):
#        points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)]
#        l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
#        l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
#        l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
#        # duplicates part
#        l4 = Line2D((2.5, 6.), (3.5, 2.)) # y = -4x + 16
#        l5 = Line2D((2., 2.5), (6., 4.5)) # y = 0.5x + 1.5
#        l6 = Line2D((3., 5.), (5., 6.)) # y = 0.5x + 3.5
#        # lines outside of point set
#        # above
#        l7 = Line2D((0., 7.), (1., 10.)) # y = 3 x + 7
#        # below
#        l8 = Line2D((0., -1.), (6., 1.)) # y = 1/3 x - 1
#        # line between points omit it
#        l9 = Line2D((3.,6.),(5.,7.))
#        lines = [l1, l2, l3, l4, l5, l6, l7, l8, l9]
#        result = preprocess_lines(lines, points)
#        self.assertEqual(len(result), 3)
#        self.assertFalse(l7 in result)
#        self.assertFalse(l8 in result)
#        self.assertFalse(l9 in result)
#        self.assertTrue(l1 in result or l4 in result)
#        self.assertTrue(l2 in result or l5 in result)
#        self.assertTrue(l3 in result or l6 in result)
#
#
#class GraphTestCase(unittest.TestCase):
#    def setUp(self):
#        self.points = range(4)
#        self.graph = mwsolv.create_graph(self.points)
#
#    def test_connected_components(self):
#        self.assertFalse(self.graph.in_same_connected_component(1,2))
#        self.assertTrue(self.graph.in_same_connected_component(1,1))
#
#    def test_adjacent(self):
#        self.assertTrue(self.graph.is_adjacent(1,2))
#        self.assertFalse(self.graph.is_adjacent(1,5))
#        adjacent_vertices = self.graph.get_adjacent_vertices(1)
#        adjacent_vertices.sort()
#        self.assertEqual(adjacent_vertices, [0,2,3])
#
#    def test_get_edges(self):
#        edges = self.graph.get_edges()
#        edges.sort()
#        expected_edges = [(0,1),(0,2),(0,3), (1,2), (1,3), (2,3)]
#        self.assertEqual(edges, expected_edges)
#
#    def test_merge_connected_components(self):
#        self.graph.merge_connected_components([0],[1])
#        edges = self.graph.get_edges()
#        edges.sort()
#        expected_edges = [(0,1),(0,2),(0,3), (1,2), (1,3), (2,3)]
#        expected_edges.remove((0,1))
#        self.assertEqual(edges, expected_edges)
#
#    def test_merge_cc_with_vertices(self):
#        self.graph.merge_cc_with_vertics(0,1)
#        self.graph.merge_cc_with_vertics(0,2)
#        self.graph.merge_cc_with_vertics(3,1)
#        edges = self.graph.get_edges()
#        edges.sort()
#        self.assertEqual(edges, [])
#
#    def test_get_connected_component(self):
#        self.graph.merge_connected_components([0],[1])
#        connected_component = self.graph.get_connected_component(1)
#        connected_component.sort()
#        self.assertEqual(connected_component, [0,1])
#
#    @unittest.expectedFailure
#    def test_cannot_find_connected_component(self):
#        connected_component = self.graph.get_connected_component(5)
#
#class MultWeightsSolvingTestCase(unittest.TestCase):
#    def setUp(self):
#        self.points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)]
#        l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
#        l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
#        l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
#        self.lines = [l1, l2, l3]
#
#    def test_solution(self):
#        solution = mwsolv.compute_spanning_tree(self.points, self.lines)
#        self.assertEqual(len(solution), 4)
#        self.assertTrue(((3., 6.), (5.,7.)) in solution)
#        self.assertTrue(((2., 2.), (6.,4.)) in solution)
#        self.assertTrue(((3., 6.), (4.25,5.)) in solution or\
#                ((4.25,5.), (5., 7.)) in solution)
#        self.assertTrue(((4.25, 5.), (6.,4.)) in solution)
#
#
#class ConnectedComponentsTestCase(unittest.TestCase):
#    def test_results_one_connected_component(self):
#        points = [ (0.,3.), (3.,4.), (9., 10.), (7.,8.), (5., 6.), (2., 1.)]
#        edges = [ ((3., 4.), (0., 3.)),
#                  ((2., 1.), (3., 4.)),
#                  ((3., 4.), (5., 6.)),
#                  ((7., 8.), (5., 6.)),
#                  ((9., 10.), (7., 8.)) ]
#        (ccs, ccs_edges) = slpsolv.connected_components(points, edges)
#        self.assertEqual(len(ccs), 1)
#        self.assertEquals(len(ccs_edges), 1)
#        cc = ccs[0]
#        self.assertItemsEqual(cc, points)
#        cc_edges = ccs_edges[0]
#        print cc_edges
#        for (p,q) in edges:
#            if p < q:
#                edge = (p,q)
#            else:
#                edge = (q,p)
#            self.assertTrue(edge in cc_edges, "%s not in %s" % (edge, cc_edges))
#
#    """def test_results_two_connected_components(self):
#        points = [ (0.,3.), (3.,4.), (9., 10.), (7.,8.), (5., 6.), (2., 1.)]
#        edges = [ ((3., 4.), (0., 3.)),
#                  ((2., 1.), (3., 4.)),
#                  #((3., 4.), (5., 6.)), now two connected components
#                  ((7., 8.), (5., 6.)),
#                  ((9., 10.), (7., 8.)) ]
#        (ccs, ccs_edges) = slpsolv.connected_components(points, edges)
#        print ccs_edges
#        self.assertEqual(len(ccs), 2)
#        self.assertEqual(len(ccs_edges), 2)
#        c1 = [ (0.,3.), (3.,4.), (2., 1.)]
#        c1_edges = [((0.,3.), (3.,4.)), ((2., 1.), (3., 4.))]
#        c2 = [ (9., 10.), (7.,8.), (5., 6.)]
#        c2_edges = [((5., 6.), (7., 8.)), ((7., 8.),(9., 10.))]
#        if (0., 3.) in ccs[0]:
#            self.assertItemsEqual(ccs[0], c1)
#            self.assertItemsEqual(ccs_edges[0], c1_edges)
#            self.assertItemsEqual(ccs[1], c2)
#            self.assertItemsEqual(ccs_edges[1], c2_edges)
#        elif (0., 3.) in ccs[1]:
#            self.assertItemsEqual(ccs[1], c1)
#            self.assertItemsEqual(ccs_edges[1], c1_edges)
#            self.assertItemsEqual(ccs[0], c2)
#            self.assertItemsEqual(ccs_edges[0], c2_edges)
#        else:
#            self.fail()
#"""
#class SarielsLPSolvingTestCase(unittest.TestCase):
#    def setUp(self):
#        self.points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)]
#        l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
#        l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
#        l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
#        self.lines = [l1, l2, l3]
#
#    def test_solution(self):
#        solution = slpsolv.compute_spanning_tree(self.points, self.lines)
#        print solution
#        self.assertEqual(len(solution), 4)
#        self.assertTrue(((3., 6.), (5.,7.)) in solution)
#        self.assertTrue(((2., 2.), (6.,4.)) in solution)
#        self.assertTrue(((3., 6.), (4.25,5.)) in solution or\
#                ((4.25,5.), (5., 7.)) in solution)
#        self.assertTrue(((4.25, 5.), (6.,4.)) in solution)
#
#class FeketeLPSolvingTestCase(unittest.TestCase):
#    def setUp(self):
#        self.points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)]
#        l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
#        l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
#        l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
#        self.lines = [l1, l2, l3]
#
#    def test_solution(self):
#        solution = flpsolv.compute_spanning_tree(self.points, self.lines)
#        print solution
#        self.assertEqual(len(solution), 4)
#        self.assertTrue(((3., 6.), (5.,7.)) in solution)
#        self.assertTrue(((2., 2.), (6.,4.)) in solution)
#        self.assertTrue(((3., 6.), (4.25,5.)) in solution or\
#                ((4.25,5.), (5., 7.)) in solution)
#        self.assertTrue(((4.25, 5.), (6.,4.)) in solution)
#
#    def test_subsets(self):
#        self.points = range(1,4)
#        subsets_gen = flpsolv.nonempty_subsets(self.points)
#        subsets = [s for s in subsets_gen]
#        print subsets
#        self.assertEqual(len(subsets), 2**(len(self.points)) - 1 -1)
#
#class OptSolverTestCase(unittest.TestCase):
#    def setUp(self):
#        self.points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)]
#        l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
#        l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
#        l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
#        self.lines = [l1, l2, l3]
#
#    def test_solution(self):
#        solution = opt_solver.compute_spanning_tree(self.points, self.lines)
#        print solution
#        self.assertEqual(len(solution), 4)
#        self.assertTrue(((3., 6.), (5.,7.)) in solution)
#        self.assertTrue(((2., 2.), (6.,4.)) in solution)
#        self.assertTrue(((3., 6.), (4.25,5.)) in solution or\
#                ((4.25,5.), (5., 7.)) in solution)
#        self.assertTrue(((4.25, 5.), (6.,4.)) in solution)


if __name__ == '__main__':
    unittest.main()
