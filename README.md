# maxOPF simulation
## Main files:

* **instance.py**: contain classes that define an instance and solution. It also contains functions that generate random instances.

* **s_maxOPF_algs.py**: Algorithms solving the simplifies problem (denoted by sMaxOPF).

* **OPF_algs.py**: Algorithms solving Optimal Power Flow (OPF). Functions here call Gurobi solver.

* **util.py**: contain helper functions such as print an instance, tune Gurobi parameters...etc.

* **test.py:** quick and dirty test.

* **sim.py:** Contain main simulation functions that store results in directory /results

* **sim_plot.py**: functions that produce figures from simulations dump files.


## Directories:
* **test-feeders/**: contain one-line diagrams as well as related papers from the literature
* **results/** pdf: figures and raw simulations data

