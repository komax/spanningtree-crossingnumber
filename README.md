Spanning Tree with Low Crossing Number
===========================

Repository for various experiments for finding a Spanning Tree with Low Crossing Number:
input: set of points P (n=|P|) [and optional a set of lines L]
output: set of edges F spans a tree for P with crossing number t = max crossings of F with one line in L


Kinds of experiments
======================
1. Input data:
  - points in a grid
  - set of points sampled uniformly

2. Set of lines:
  - generating all possible lines fromed by equivalent classes on each pair of points p,q in P
  - lines are created by a sample following different distribuitions

3. Algortihms for finding a spanning tree:
  - exact solution: integer programming, brute-force (all combinations)
  - approxmation:
    a. multiplicative weights on lines and edges
    b. linear program with relaxation and rounding scheme (determinstic in the plane)

Used infrastructure
======================
* Python
* matlab (for plotting)
* Gurobi (for solving LP, IP)
