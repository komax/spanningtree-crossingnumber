''' unit test (suite) module checks functionality of all other programmed modules '''

import unittest
import mult_weights_solver as mwsolv
import datagenerator as dgen
from datagenerator import Line2D, LineSegment2D

class CrossingTestCase(unittest.TestCase):
    def test_has_crossing(self):
        line = Line2D((3,2), (5,0)) # y = -x + 5
        line_segment = LineSegment2D((3,0.5), (5,1.5)) # y = 0.5 x - 1
        self.assertTrue(mwsolv.has_crossing(line, line_segment))

    def test_has_no_crossing_linesegment_too_short(self):
        line = Line2D((3,2), (5,0)) # y = -x + 5
        line_segment = LineSegment2D((0,-1), (3,0.5)) # y = 0.5 x - 1
        self.assertFalse(mwsolv.has_crossing(line, line_segment))

    def test_has_no_crossing_line_and_segment_parallel(self):
        line = Line2D((3,2), (5,0)) # y = -x + 5
        line_segment = LineSegment2D((0, 4), (6, -2)) # y = -x + 4
        self.assertFalse(mwsolv.has_crossing(line, line_segment))

class GraphTestCase(unittest.TestCase):
    def setUp(self):
        self.points = range(4)
        self.graph = mwsolv.create_graph(self.points)

    def test_connected_components(self):
        self.assertFalse(self.graph.in_same_connected_component(1,2))
        self.assertTrue(self.graph.in_same_connected_component(1,1))

    def test_adjacent(self):
        self.assertTrue(self.graph.is_adjacent(1,2))
        self.assertFalse(self.graph.is_adjacent(1,5))
        adjacent_vertices = self.graph.get_adjacent_vertices(1)
        adjacent_vertices.sort()
        self.assertEqual(adjacent_vertices, [0,2,3])

    def test_get_edges(self):
        edges = self.graph.get_edges()
        edges.sort()
        expected_edges = [(0,1),(0,2),(0,3), (1,2), (1,3), (2,3)]
        self.assertEqual(edges, expected_edges)

    def test_merge_connected_components(self):
        self.graph.merge_connected_components([0],[1])
        edges = self.graph.get_edges()
        edges.sort()
        expected_edges = [(0,1),(0,2),(0,3), (1,2), (1,3), (2,3)]
        expected_edges.remove((0,1))
        self.assertEqual(edges, expected_edges)

    def test_merge_cc_with_vertices(self):
        self.graph.merge_cc_with_vertics(0,1)
        self.graph.merge_cc_with_vertics(0,2)
        self.graph.merge_cc_with_vertics(3,1)
        edges = self.graph.get_edges()
        edges.sort()
        self.assertEqual(edges, [])

    def test_get_connected_component(self):
        self.graph.merge_connected_components([0],[1])
        connected_component = self.graph.get_connected_component(1)
        connected_component.sort()
        self.assertEqual(connected_component, [0,1])

    @unittest.expectedFailure
    def test_cannot_find_connected_component(self):
        connected_component = self.graph.get_connected_component(5)

class MultWeightsSolvingTestCase(unittest.TestCase):
    def setUp(self):
        self.points = [(2.,2.), (6.,4.), (3., 6.), (5., 7.), (4.25, 5.)]
        l1 = Line2D((2., 6.), (3., 2.)) # y = -4x + 14
        l2 = Line2D((2., 3.), (6., 5.)) # y = 0.5x + 2
        l3 = Line2D((3., 5.5), (5., 6.5)) # y = 0.5x + 4
        self.lines = [l1, l2, l3]

    def test_solution(self):
        solution = mwsolv.compute_spanning_tree(self.points, self.lines)
        self.assertEqual(len(solution), 4)
        self.assertTrue(((3., 6.), (5.,7.)) in solution)
        self.assertTrue(((2., 2.), (6.,4.)) in solution)
        self.assertTrue(((3., 6.), (4.25,5.)) in solution or\
                ((5., 7.), (4.25,5.)) in solution)
        # TODO why does this assert fails?
        self.assertTrue(((4.25, 5.), (6.,4.)) in solution)



if __name__ == '__main__':
    unittest.main()
