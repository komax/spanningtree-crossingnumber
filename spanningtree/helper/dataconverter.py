'''
Created on Mar 6, 2013

@author: max
'''

import sys

assert 3 >= len(sys.argv) >= 2
infilename = sys.argv[1]
assert infilename.endswith('.tsp')

if len(sys.argv) == 3:
    outfilename = sys.argv[2]
    sys.stdout = open(outfilename, 'w')

with open(infilename, 'rb') as tspfile:
    found_data_part = False
    for line in tspfile:
        if 'EOF' in line:
            break
        elif found_data_part:
            splitted = line.split()
            # no node counting
            new_line = ' '.join(splitted[1:])
            print new_line
        elif 'NODE_COORD_SECTION' in line:
            found_data_part = True
            continue
