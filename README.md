# Spanning Tree with Low Crossing Number
This Repository implements experiments for finding a [Spanning Tree with Low Crossing Number](
http://arxiv.org/abs/cs/0310034)
- The input is: set of points P (n=|P|) and optional a set of lines L
- An algorithm for this problem outputs: set of edges F forming a spanning tree for P with crossing number t
- t is the maximum crossing number of any line in L

This project is released under the MIT license.

## Installation/Prerequisites
* Python 2.7
* [NumPy](http://www.numpy.org/) for efficient computation
* [matplotlib](http://matplotlib.org/) for plotting
* [Gurobi](http://www.gurobi.com/) solving LPs

1. Install the above libraries as they are not preinstalled


    $ sudo apt-get install python-numpy python-matplotlib


2. Check if the current Gurobi license is activated. A correct prompt looks
   like this


    $ gurobi.sh 
    Python 2.7.2 (default, Nov 21 2011, 12:59:35) 
    [GCC 4.2.4 (Ubuntu 4.2.4-1ubuntu4)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.

    Gurobi Interactive Shell (linux64), Version 5.1.0
    Copyright (c) 2013, Gurobi Optimization, Inc.
    Type "help()" for help

    gurobi> 


3. Clone the repository

    $ git clone https://github.com/komax/spanningtree-crossingnumber


## Types of Experiments
Two command line tools were developed:
1. Executation of a single experiment
2. Generation of csv files

An experiments consists of these parameters
* generation of uniform or grid data
* reading data from a file
* controlling sampling of lines
* selecting a solver
* specification of the output

The outputs for an single experiment are
* Standard Output in form of text information
* dumping the graph as png file
* open the GUI from matplotlib to browse the spanning tree

The csv tool dumps only the text information (crossing number, size of the
points, number of lines, ...)

## Usage
An experiment has the following flags:

    $ ./experiment.py -h
    usage: spanning_tree_with_low_crossing.py [-h]
                                              [-s {hp_lp,fekete_lp,cc_lp,mult_weight,opt,all}]
                                              [-d DIMENSIONS] [-n NUMBER]
                                              [-g {uniform,grid}] [-i INPUT]
                                              [-l {all,stabbing,random}] [-m MEAN]
                                              [-p] [-r] [-v]

    run an parameterized experiment to compute a spanning tree with low crossing
    number

    optional arguments:
      -h, --help            show this help message and exit
      -s {hp_lp,fekete_lp,cc_lp,mult_weight,opt,all}, --solver {hp_lp,fekete_lp,cc_lp,mult_weight,opt,all}
                            choose an algorithm computing a feasible solution
      -d DIMENSIONS, --dimensions DIMENSIONS
                            number of dimensions of point set
      -n NUMBER, --number NUMBER
                            quantify how many points you want
      -g {uniform,grid}, --generate {uniform,grid}
                            how should the point set be sampled
      -i INPUT, --input INPUT
                            specify a input file
      -l {all,stabbing,random}, --linesampling {all,stabbing,random}
                            structure of the line set
      -m MEAN, --mean MEAN  run experiment m times and returns averaged results
      -p, --plot            plots the computed solution into a widget
      -r, --record          record computed solution into a png
      -v, --verbose         adds verbose outputs to STDOUT


An example usage is

    $  ./experiment.py -n 16 -g grid -l stabbing -s mult_weight -r -v
    Start generating graph...
    Graph has been created.
    Sampling of lines started...
    Sampling of lines finished.
    Start now computing a spanning tree...
    Computing a spanning tree finished.
    CPU time (in sec) 2.64830684662
    crossing number=5
    iterations=15
    number of points=16
    number of lines=120
    all crossings=330
    average crossing number=2.75
    Start now plotting...
    Closed plot.

The flags the flags for generating a csv file are almost the same, except the
specification for range 4..20 and stepping by `--increment 2`:
    $ ./csv_experiment.py -f 4 -t 20 --increment 2

``-c`` generates a human readable header prepending the csv file. Use this flag
for understanding the file format or consult the generated results in the
directory `results`
