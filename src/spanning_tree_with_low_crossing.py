#! /usr/bin/env python
''' main module to read files, start, evaluate and output different experiments
  See -h for further information
'''

import argparse
import time
import copy
import datagenerator as dgen
import mult_weights_solver as mws
import sariel_lp_solver as slpsolv
import fekete_lp_solver as flpsolv
import opt_solver
from lines import crossing_tuple, preprocess_lines
import plotting

def main():
    args = parse_args()
    experiment = prepare_experiment(args)
    experiment.run()
    experiment.print_results()
    experiment.plot()

# constants for options
SOLVER_OPTIONS = ['opt', 'mult_weight', 'fekete_lp', 'sariel_lp']
DATA_DISTRIBUTION_OPTIONS = ['uniform', 'grid']
NO_LINES_SAMPLING = -1

def parse_args():
    '''
    parse all arguments on the command line and return them
    '''
    parser = argparse.ArgumentParser(description=
            """
            run an parameterized experiment to compute
            a spanning tree with low crossing number
            """)
    parser.add_argument("-s", "--solver", default='mult_weight',
            choices=SOLVER_OPTIONS,
            help="choose an algorithm computing a feasible solution")
    parser.add_argument("-d", "--dimensions", type=int, default=2,
        help="number of dimensions of point set")
    parser.add_argument("-n", "--number", type=int, default=16,
        help="quantify how many points you want")
    parser.add_argument("-g", "--generate", default='uniform',
            choices=DATA_DISTRIBUTION_OPTIONS,
        help="how should the point set be sampled")
    parser.add_argument("-l", "--linessize", type=int,
            default=NO_LINES_SAMPLING,
        help="quantify how many random lines you want")
    parser.add_argument("-p", "--plot", action='store_true', default=False,
        help="plots the computed solution into a widget")
    parser.add_argument("-v", "--verbose", action='store_true',
        help="adds verbose outputs to STDOUT")
    args = parser.parse_args()
    return args

def prepare_experiment(args):
    '''
    factory for creating a SpanningTreeExperiment from arguments
    '''
    return SpanningTreeExperiment(args.solver, args.dimensions, args.number,
            args.generate, args.linessize, args.plot, args.verbose)

def generate_point_set(d, n, distribution_type):
    '''
    generic sampling function for different point sets (dimensions,
    distributions)
    '''
    # TODO currently omitting d parameter. update it
    assert distribution_type in DATA_DISTRIBUTION_OPTIONS
    if distribution_type == 'uniform':
        return dgen.generate_points_uniformly(n)
    elif distribution_type == 'grid':
        return dgen.generate_points_grid(n)

def get_solver(solver_type):
    '''
    select a solver by solver_type and use return solver function later
    '''
    # TODO update if new solvers are supported
    assert solver_type in SOLVER_OPTIONS
    if solver_type == 'mult_weight':
        return mws.compute_spanning_tree
    elif solver_type == 'sariel_lp':
        return slpsolv.compute_spanning_tree
    elif solver_type == 'fekete_lp':
        return flpsolv.compute_spanning_tree
    elif solver_type == 'opt':
        return opt_solver.compute_spanning_tree
    else:
        raise StandardError('Not yet supported this |%s| solver type' %
                solver_type)

class SpanningTreeExperiment:
    '''
    stores all necessary information, data and options to preprocess point or
    line set before running a solver and takes care of time measuring
    '''
    def __init__(self, solver_type, d, n, distribution_type, linessize, has_plot, verbose):
        self.points = generate_point_set(d, n, distribution_type)
        if linessize == NO_LINES_SAMPLING:
            lines = dgen.generate_lines(self.points)
        else:
            lines = dgen.generate_random_lines(linessize, self.points)
        self.lines = preprocess_lines(lines, self.points)
        self.solver_type = solver_type
        self.solver = get_solver(solver_type)
        self.has_plot = has_plot
        self.verbose = verbose

        self.elapsed_time = None
        self.solution = None
        self.crossing_number = None

    def update_solver(self, solver_type):
        '''
        exchange a solver in the experiment with a new one
        '''
        self.solver_type = solver_type
        self.solver = get_solver(solver_type)

    def run(self):
        '''
        compute a spanning tree and takes care of
        caching of points, lines, time mearsuring and storing the crossing
        number
        '''
        points = copy.deepcopy(self.points)
        lines = copy.deepcopy(self.lines)
        start = time.time()
        self.solution = self.solver(points, lines)
        end = time.time()
        self.elapsed_time = end - start
        (min_crossing_number, crossing_number, crossings) = \
            crossing_tuple(self.lines, self.solution)
        self.min_crossing_number = min_crossing_number
        self.crossing_number = crossing_number
        self.crossings = crossings

    def print_results(self):
        '''
        print all important statistics to STDOUT: crossing number, elapsed
        CPU time. If verbose flag is set:
         - number of points
         - number of lines,
         - minimum crossing number,
         - sum of all crossings and
         - average crossing number
        '''
        print "CPU time (in sec) %s" % self.elapsed_time
        print "crossing number=%s" % self.crossing_number
        if self.verbose:
            print "number of points=%s" % len(self.points)
            no_lines = len(self.lines)
            print "number of lines=%s" % no_lines
            print "minimum crossing number=%s" % self.min_crossing_number
            print "all crossings=%s" % self.crossings
            average_crossing_number = float(self.crossings) / no_lines
            print "average crossing number=%s" % average_crossing_number

    def plot(self):
        '''
        if plot option is set, plot
        '''
        if self.has_plot:
            plotting.plot(self.points, self.lines, self.solution)
        return

if __name__ == '__main__':
    main()
