#! /usr/bin/env python
import os
import sys

args = sys.argv[1:]
os.system('python -m spanningtree.csv_experiment_statistics '+
        ' '.join(args))
