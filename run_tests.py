#! /usr/bin/env python
import os
import sys

args = sys.argv[1:]
os.system('python -m spanningtree.test.unittests'+
        ' '.join(args))
