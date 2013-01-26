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
from spanning_tree_with_low_crossing_number import SpanningTreeExperiment
from spanning_tree_with_low_crossing_number import SOLVER_OPTIONS
from spanning_tree_with_low_crossing_number import DATA_DISTRIBUTION_OPTIONS
from spanning_tree_with_low_crossing_number import NO_LINES_SAMPLING

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
    parser.add_argument('-o', '--out', metavar='file',
            default='experiments.csv', type=argparse.FileType('wb'))
    parser.add_argument("-s", "--solver", default='mult_weight',
            choices=SOLVER_OPTIONS,
            help="choose an algorithm computing a feasible solution")
    parser.add_argument("-g", "--generate", default='uniform',
            choices=DATA_DISTRIBUTION_OPTIONS,
        help="how should the point set be sampled")
    parser.add_argument("-f", "--begin", type=int, default=4,
        help="starting with how many points")
    parser.add_argument("-t", "--to", type=int, default=16,
        help="ending up with how many points")
    parser.add_argument("-i", "--increment", type=int, default=1,
        help="number of steps in the size of point set for next experiment")
    args = parser.parse_args()
    return args

def prepare_experiment(args):
    '''
    factory method creating a CompoundExperiment object out of argparse
    arguments
    '''
    return CompoundExperiment(args.out, args.solver, args.generate, args.begin,
            args.to, args.increment)

class CompoundExperiment:
    def __init__(self, file_name, solver_type, distribution_type, lb, ub,
            step):
        self.file_name = file_name
        dimension = 2
        has_plot = False
        verbose = False
        self.distribution_type = distribution_type
        self.lb = lb
        self.ub = ub
        stelf.step = step
        self.experiment = SpanningTreeExperiment(solver_type, dimension, lb,
                distribution_type, NO_LINES_SAMPLING, has_plot, verbose)

   def write_csv(self):
       with open(self.file_name, 'wb') as csvfile:
           csv_writer = csv.writer(csvfile)
           if self.distribution_type == 'grid':
               lb = int(math.sqrt(self.lb))
               ub = int(math.sqrt(self.ub))
           else:
               lb = self.lb
               ub = self.ub
           dimension = 2
           for i in range(lb, ub, step):
               self.experiment.clean_up()
               if self.distribution_type == 'grid':
                   i = i**2

               self.experiment.update_point_set(dimension, i,
                       self.distribution_type)
               self.experiment.run()
               results = self.experiment.results()
               csv_writer.writerow(results)


if __name__ == '__main__':
    main()
