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
from spanning_tree_with_low_crossing import SOLVER_OPTIONS
from spanning_tree_with_low_crossing import DATA_DISTRIBUTION_OPTIONS
from spanning_tree_with_low_crossing import LINE_OPTIONS
from spanning_tree_with_low_crossing import create_experiment

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
    parser.add_argument("-d", "--dimensions", type=int, default=2,
        help="number of dimensions of point set")
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
    parser.add_argument("-m", "--mean", type=int, default=1,
        help="run experiment m times and return an averaged result")
    parser.add_argument("-v", "--verbose", action='store_true',
        help="adds verbose outputs to STDOUT")
    args = parser.parse_args()
    return args

def prepare_experiment(args):
    '''
    factory method creating a CompoundExperiment object out of argparse
    arguments
    '''
    return CompoundExperiment(args.out, args.solver, args.dimensions,
                              args.generate,
                              args.linesampling, args.begin,
                              args.to, args.increment, args.csvheader,
                              args.mean, args.verbose)

csv_header = [
        'size of point set',
        'size of line set',
        'CPU time in seconds',
        'iterations',
        'crossing number',
        'minimum crossing number',
        'average crossing number',
        'crossing overall']


class CompoundExperiment:
    def __init__(self, file_name, solver_type, dimensions, distribution_type, line_option, lb, ub,
            step, has_header, mean, verbose):
        self.file_name = file_name
        has_plot = False
        self.solver_type = solver_type
        self.verbose = verbose
        self.distribution_type = distribution_type
        self.line_option = line_option
        self.lb = lb
        self.ub = ub
        self.step = step
        self.has_header = has_header
        self.experiment = create_experiment(solver_type, dimensions, lb,
                                            distribution_type,
                                            line_option, mean, has_plot, verbose)
        
    def write_csv(self):
        self.write_header()
        self.write_data()
    
    def write_header(self):
        if self.has_header:
            for file_name in self.get_files():
                with open(file_name, 'wb') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(csv_header)
                    
    
    def get_files(self):
        if self.solver_type == 'all':
            return [self.file_name+'.'+solv_ext for solv_ext in SOLVER_OPTIONS[:-1]]
        else:
            return [self.file_name] 

    def write_data(self):
        if self.distribution_type == 'grid':
            lb = int(math.sqrt(self.lb))
            ub = int(math.sqrt(self.ub))
        else:
            lb = self.lb
            ub = self.ub
        dimension = 2
        for i in range(lb, ub+1, self.step):
            if self.distribution_type == 'grid':
                i = i**2
            if self.verbose:
                print "Starting now a new experiment for n=%s..." % i
            if self.solver_type == 'all':
                for experiment in self.experiment:
                    experiment.update_point_set(dimension, i,
                                                self.distribution_type,
                                                self.line_option)
                    experiment.run()
                    results = experiment.results()
                    if self.verbose:
                        print "Computation finished. Writing results to csv..."
                    with open(self.file_name+'.'+experiment.solver_type, 'ab') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(results)
            else:
                self.experiment.run()
                results = self.experiment.results()
                with open(self.file_name, 'ab') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(results)
        return

if __name__ == '__main__':
    main()
