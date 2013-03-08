#! /usr/bin/env python
import os
import sys

args = sys.argv[1:]
os.system('python -m spanningtree.spanning_tree_with_low_crossing'+
        ' '.join(args))
