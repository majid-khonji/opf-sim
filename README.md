# maxOPF simulation
There are three main files:

* instance.py: contain classes that define an instance and solution. It also contains functions that generate random instances.

* s_maxOPF_algs.py: Algorithms solving the simplifies problem (denoted by sMaxOPF).

* OPF_algs.py: Algorithms solving MaxOPF and minimum loss Optimal Power Flow (OPF). Functions here call Gurobi solver to perform the work load.

* util.py: contain helper functions such as print an instance, tune Gurobi parameters...etc.

* test.py: quick and dirty test.

* sim.py: Contain main simulation functions sim().

* sim_plot.py: functions that produce figures from simulations dump files.



