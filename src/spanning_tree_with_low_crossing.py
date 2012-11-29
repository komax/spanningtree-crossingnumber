''' main module to read files, start, evaluate and output different experiments
  See -h for further information
'''

import argparse
import time

def main():
    args = parse_args()
    procees_args(args)
    time_measuring_of_experiment()

solver_options = ['opt', 'mult_weight', 'fekete_lp', 'sariel_lp']
generated_data_options = ['uniform', 'grid']

def parse_args():
    parser = argparse.ArgumentParser(description=
            """
            run an parameterized experiment to compute
            a spanning tree with low crossing number
            """)
    parser.add_argument("-d", "--dimension", type=int, default=2,
        help="number of dimensions of point set")
    parser.add_argument("-s", "--solver", default='opt', choices=solver_options,
        help="choose an algorithm computing a feasible solution")
    parser.add_argument("-g", "--generate", default='uniform',
            choices=generated_data_options,
        help="how should the point set be sampled")
    parser.add_argument("-n", "--number", type=int, default=16,
        help="quantify how many points you want")
    parser.add_argument("-p", "--plot", action='store_true', default=True,
        help="plots the computed solution into a widget")
    parser.add_argument("-v", "--verbose", action='store_true',
        help="adds verbose outputs to STDOUT")
    args = parser.parse_args()
    return args

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
