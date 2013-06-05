"""
generates a diagram to compare some results (from csv) in a 2D chart
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

algorithm_names = [
        'mult weights',
        'fekete lp',
        'har-peled lp',
        'connected comp lp'
        ]

markers = [
        'r-', # red line
        'b-',  # blue line
        'g-',  # green line
        'y-'   # yellow line
        ]

def plot_chart(filename, csv_files):
    plt.figure()
    filename = filename + '.png'
    # TODO add legend
    for (i, csv_file) in enumerate(csv_files):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        my_marker = markers[i]
        #plt.plot(data[:, 0], data[:, 4], my_marker)
        plt.plot(data[:, 0], data[:, 2], my_marker)
        #plt.plot(data[:, 0], data[:, 6], my_marker)
    points = np.arange(2, 101)
    plt.plot(points, 0.5*points**2, 'k--')
    plt.xlabel('number of points')
    plt.ylabel('CPU time (in sec)')
    plt.savefig(filename)
    plt.show()

def main():
    assert len(sys.argv) >= 3
    filename = sys.argv[1]
    csv_files = sys.argv[2:]
    plot_chart(filename, csv_files)

if __name__ == '__main__':
    main()
