"""
generates a diagram to compare some results (from csv) in a 2D chart
"""

import matplotlib.pyplot as plt
import pylab
import numpy as np
import sys

legend_names = [
#        'mult weights',
        'fekete lp',
#        'har-peled lp (20 runs)'
        'connected comp lp'
        ]

markers = [
#        'r-', # red line
        'b-',  # blue line
#        'g-',  # green line
        'y-'   # yellow line
        ]

def plot_chart(filename, csv_files):
    plt.figure()
    filename = filename + '.png'
    for (i, csv_file) in enumerate(csv_files):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        my_marker = markers[i]
        my_label = legend_names[i]
        # x: n y: crossing number
        #plt.plot(data[:, 0], data[:, 4], my_marker, label=my_label)
        # x: n y: cpu time
        #plt.plot(data[:, 0], data[:, 2], my_marker, label=my_label)
        # x: n y: average crossing number
        plt.plot(data[:, 0], data[:, 5], my_marker, label=my_label)
        # for fekete trails
        # x: n y: failed trails
        #plt.plot(data[:, 0], data[:, -1], my_marker, label=my_label)
    points = np.arange(2, 101)
    #plt.plot(points, 4*points, 'k--', label='$f(x)=4x$')
    #plt.plot(points, points*3, 'k--', label='$f(x)=3x$')
    plt.plot(points, np.sqrt(points), 'k--', label='$f(x)=\sqrt{x}$')
    #plt.plot(points, np.log(points), 'k--', label='$f(x)=\log(x)$')
    plt.xlabel('number of points')
    #plt.ylabel('average trails adding a heavy weight edge within a cc')
    plt.ylabel('average crossing number')
    #plt.ylabel('crossing number')
    pylab.legend(loc=0)
    plt.savefig(filename)
    plt.show()

def main():
    assert len(sys.argv) >= 3
    filename = sys.argv[1]
    csv_files = sys.argv[2:]
    plot_chart(filename, csv_files)

if __name__ == '__main__':
    main()
