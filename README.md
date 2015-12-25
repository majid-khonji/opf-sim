# maxOPF simulation
There are three main files:

* instance.py (reusable code):__ contain all algorithms (greedy ratio/demand/ valuation and Gurobi OPT invokation). These function receive an object of class ckp_instance and return an object of class ckp_sol. To generate a simple random instance you can invoke ckp_instance_rnd()

* s_maxOPF_algs.py:

* OPF_algs.py:

* util.py:

* test.py: quick and dirty test

* sim.py: Contain main simulation functions sim(). The input of these functions is the scenario as defined in "power/demand-response/draft5" with full communication. These sim function invoke another local function called ckp_instance_sim() which generates a single instance bases on a given scenario and system configs. Once a simulation is run, it outputs results in "results/dump". The files in this directory contain the figures data (with confidence interval calculated. The next file uses the output of these files.

* ckp_sim_plot.py: main plot functions are plot_all(type="[obj | cr | cr_pj]") ("cr" stands for competitive ratio while actually we meant approximation ratio, we will fix that later.); plot_time() for time figures; and plot_dyn_C(). Remaining functions are for formating.



