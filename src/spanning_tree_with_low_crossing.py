''' main module to read files, start, evaluate and output different experiments
  See -h for further information
'''

import argparse
import time
import datagenerator as dgen
import mult_weights_solver as mws
import sariel_lp_solver as slpsolv

def main():
    args = parse_args()
    procees_args(args)
    time_measuring_of_experiment()

solver_options = ['opt', 'mult_weight', 'fekete_lp', 'sariel_lp']
data_distribution_options = ['uniform', 'grid']

def parse_args():
    parser = argparse.ArgumentParser(description=
            """
            run an parameterized experiment to compute
            a spanning tree with low crossing number
            """)
    parser.add_argument("-s", "--solver", default='opt', choices=solver_options,
        help="choose an algorithm computing a feasible solution")
    parser.add_argument("-d", "--dimension", type=int, default=2,
        help="number of dimensions of point set")
    parser.add_argument("-n", "--number", type=int, default=16,
        help="quantify how many points you want")
    parser.add_argument("-g", "--generate", default='uniform',
            choices=data_distribution_options,
        help="how should the point set be sampled")
    parser.add_argument("-p", "--plot", action='store_true', default=True,
        help="plots the computed solution into a widget")
    parser.add_argument("-v", "--verbose", action='store_true',
        help="adds verbose outputs to STDOUT")
    args = parser.parse_args()
    return args

def generate_point_set(d, n, distribution_type):
    # TODO currently omitting d parameter. update it
    assert distribution_type in data_distribution_options
    point_set = []
    if distribution_type == 'uniform':
        point_set = dgen.generate_points_uniformly(n)
    elif distribution_type == 'grid':
        point_set = dgen.generate_points_grid(n)
    return point_set

def get_solver(solver_type):
    # TODO update if new solvers are supported
    assert solver_type in solver_options
    if solver_type == 'mult_weight':
        def comp_mws(points, lines):
            return mws.compute_spanning_tree(points, lines)
        return comp_mws
    elif solver_type == 'sariel_lp':
        def sariel_lp(points, lines):
            return slpsolv.compute_spanning_tree(points, lines)
        return sariel_lp
    else:
        raise StandardError('Not yet supported')

class SpanningTreeExperiment:
    def __init__(self, solver_type, d, n, distribution_type, verbose):
        self.points = generate_point_set(d, n, distribution_type)
        self.lines = dgen.generate_lines(self.points)
        self.solver_type = solver_type
        self.solver = get_solver(solver_type)
        self.verbose = verbose

        self.elapsed_time = None
        self.solution = None
        self.crossing_number = None

    def update_solver(self, solver_type):
        self.solver_type = solver_type
        self.solver = get_solver(solver_type)

    def run(self):
        start = time.time()
        self.solution = self.solver(self.points, self.lines)
        end = time.time()
        self.elapsed_time = end - start



def process_args(args):
    pass

def time_measuring_of_experiment(experiment):
    start = time.time()
    solution = experiment()
    end = time.time()
    print "CPU time (musec) %s" % (end-start)

def run_experiment():
    pass


if __name__ == '__main__':
    main()
