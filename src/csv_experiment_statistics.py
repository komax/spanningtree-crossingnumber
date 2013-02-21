#! /usr/bin/env python
'''
main module to write a csv file for experiment batch. Each line of the csv
corresponds to an experiment. Uses extensivly SpanningTreeExperiment class from
spanning_tree_with_low_crossing_number module
See -h for further information
'''

import argparse
import csv
import math
from spanning_tree_with_low_crossing import SpanningTreeExperiment
from spanning_tree_with_low_crossing import SOLVER_OPTIONS
from spanning_tree_with_low_crossing import DATA_DISTRIBUTION_OPTIONS
from spanning_tree_with_low_crossing import LINE_OPTIONS

def main():
    args = parse_args()
    experiment = prepare_experiment(args)
    experiment.write_csv()

def parse_args():
    '''
    parse all arguments on the command line and return them
    '''
    parser = argparse.ArgumentParser(description=
            """
            run a family/bunch of experiments to compute
            a spanning tree with low crossing number and write all results
            in csv file
            """)
    parser.add_argument('-o', '--out',
            default='experiments.csv', type=str)
    parser.add_argument("-c", "--csvheader", action='store_true', default=False,
        help="prepends a human readable header to csv file")
    parser.add_argument("-s", "--solver", default='mult_weight',
            choices=SOLVER_OPTIONS,
            help="choose an algorithm computing a feasible solution")
    parser.add_argument("-g", "--generate", default='uniform',
            choices=DATA_DISTRIBUTION_OPTIONS,
        help="how should the point set be sampled")
    parser.add_argument("-l", "--linesampling", default='all',
            choices=LINE_OPTIONS,
        help="structure of the line set")
    parser.add_argument("-f", "--begin", type=int, default=4,
        help="starting with how many points")
    parser.add_argument("-t", "--to", type=int, default=16,
        help="ending up with how many points")
    parser.add_argument("-i", "--increment", type=int, default=1,
        help="number of steps in the size of point set for next experiment")
    parser.add_argument("-v", "--verbose", action='store_true',
        help="adds verbose outputs to STDOUT")
    args = parser.parse_args()
    return args

def prepare_experiment(args):
    '''
    factory method creating a CompoundExperiment object out of argparse
    arguments
    '''
    return CompoundExperiment(args.out, args.solver, args.generate,
                              args.linesampling, args.begin,
                              args.to, args.increment, args.csvheader,
                              args.verbose)

csv_header = [
        'size of point set',
        'size of line set',
        'CPU time in seconds',
        'crossing number',
        'minimum crossing number',
        'average crossing number',
        'crossing overall']


class CompoundExperiment:
    def __init__(self, file_name, solver_type, distribution_type, line_option, lb, ub,
            step, has_header, verbose):
        self.file_name = file_name
        dimension = 2
        has_plot = False
        self.verbose = verbose
        self.distribution_type = distribution_type
        self.lb = lb
        self.ub = ub
        self.step = step
        self.has_header = has_header
        self.line_opton = line_option
        self.experiment = SpanningTreeExperiment(solver_type, dimension, lb,
                distribution_type, line_option, has_plot, verbose)

    def write_csv(self):
        with open(self.file_name, 'wb') as csvfile:
            csv_writer = csv.writer(csvfile)
            if self.has_header:
                csv_writer.writerow(csv_header)

            if self.distribution_type == 'grid':
                lb = int(math.sqrt(self.lb))
                ub = int(math.sqrt(self.ub))
            else:
                lb = self.lb
                ub = self.ub
            dimension = 2
            for i in range(lb, ub+1, self.step):
                self.experiment.clean_up()
                if self.distribution_type == 'grid':
                    i = i**2

                if self.verbose:
                    print "Starting now a new experiment for n=%s..." % i

                self.experiment.update_point_set(dimension, i,
                       self.distribution_type, self.line_opton)
                self.experiment.run()
                results = self.experiment.results()
                if self.verbose:
                    print "Computation finished. Writing results to csv..."
                csv_writer.writerow(results)
        return

if __name__ == '__main__':
    main()
