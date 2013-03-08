'''
Created on Mar 8, 2013

@author: max
'''
class CrossingRegistry:
    def __init__(self):
        self.crossings = {}

    @staticmethod
    def convert(line, line_seg):
        return (id(line), id(line_seg))

    def put(self, line, line_seg, bool_val):
        i, j = CrossingRegistry.convert(line, line_seg)
        self.crossings[(i, j)] = bool_val

    def has_entry(self, line, line_seg):
        i, j = CrossingRegistry.convert(line, line_seg)
        return (i, j) in self.crossings

    def has_crossing(self, line, line_seg):
        if self.has_entry(line, line_seg):
            i, j = CrossingRegistry.convert(line, line_seg)
            return self.crossings[(i, j)]
        else:
            raise StandardError('line=%s and line_seg=%s not in registry' % 
                    (line, line_seg))

registry = CrossingRegistry()

def new_crossing_registry():
    global registry
    registry = CrossingRegistry()

def has_crossing(line, line_seg):
    if registry.has_entry(line, line_seg):
        return registry.has_crossing(line, line_seg)
    else:
        cross_val = calc_has_crossing(line, line_seg)
        registry.put(line, line_seg, cross_val)
        return cross_val


def calc_has_crossing(line, line_seg):
    '''
    Has line a crossing with the line segment
    '''

    if np_allclose(line.theta[..., :-1], line_seg.theta[..., :-1]):
        return False
    else:
        A = np.vstack([line.theta, line_seg.theta])
        b = -A[..., -1:]
        A[..., -1] = -np.ones(len(A))
        intersection_point = (np.linalg.solve(A, b)).flatten()
        x = intersection_point[..., :-1]
        return line_seg.is_between(x)