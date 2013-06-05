"""
generates a diagram to compare some results (from csv) in a 2D chart
"""

import matplotlib.pyplot as plt
import numpy as np

csv_files = [
        ]

markers = [
        'r--', # red dashed line
        'bs',  # blue squares
        'g^',  # green triangles
        'yo'   # yellow circles
        ]

def plot_chart(filename):
    plt.figure()
    filename = filename + '.png'
    # TODO add legend
    for (i, csv_file) in enumerate(csv_files):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        my_marker = markers[i]
        plt.plot(data[:, 0], data[:, 4], my_marker)
    plt.xlabel('number of points')
    plt.ylabel('crossing number')
    plt.savefig(filename)
    plt.show()

def main():
    pass

if __name__ == '__main__':
    main()
