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

class EdgesTestCase(unittest.TestCase):
    def setUp(self):
        self.points = range(4)
        self.edges = mwsolv.generate_edges(self.points)

    def test_connected_components(self):
        self.assertFalse(self.edges.in_same_connected_component(1,2))
        self.assertTrue(self.edges.in_same_connected_component(1,1))

    def test_adjacent(self):
        self.assertTrue(self.edges.is_adjacent(1,2))
        self.assertFalse(self.edges.is_adjacent(1,5))
        adjacent_vertices = self.edges.get_adjacent_vertices(1)
        adjacent_vertices.sort()
        self.assertEqual(adjacent_vertices, [0,2,3])



if __name__ == '__main__':
    unittest.main()
