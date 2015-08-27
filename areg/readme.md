Simulation interface for the tree case
-------------------------------------------------------------------
The code consists of two parts: Simulator.py and main.py. In the former file, I implemented the Simulation interface and the algorithms. Whereas, the latter file contains the code for generating the tree topology and customer data set.


Simulator.py
------------------------------------------
You will need the followig libraries to be installed before running the code;

import simpy
import main
import time
import math
import networkx as nx
import gurobipy as gbp
import copy



There is one class MicroGrid which contains the simulation interface and the algorithms. Right under the class declaration you will find the simulation parameters and initiating function.
The MicroGrid class's methods are as follows;


MicroGrid

	__init__ - Initiates the simulation interface
	getOverallUtility - Returns the overall utility of all the customers
	getOverallDemand - Returns the overall demand of all the customers
	run - Initiates the maximisation procedure
	Maximize - Calls the algorithms and sets timeout for each iteration
	findDemands - Returns the set of customers sharing an edge
	GurobiOPT - Gurobi optimiser
	GreedyTree - Your Greedy algorithm
	GreedyMagnitude - Greedy Demand algorithm that is called in the GreedyTree


Parameters following the class ar self explanatory



main.py
---------------------------------------------------------------------------
You will need the followig libraries to be installed before running the code;


import math
import random
import networkx as nx

There are several function in this file which are as follows;

deletewithProbability - returns true or false (whether to delete or no)
graphGenerator - Generates the tree. Please note that the generated tree is specific and may not be suitable for your experiments. It generates tree with dept of four on each level having random number of nodes (except the first level where it has nly one node). Also, assigns the edge capacities, which are calculating by divding the capacity of the ingoing edge of a node by the number of outgoing edges from a node (except the first edge which is 5MW) 
constrained_sum_sample_pos - Returns a randomly chosen list of n positive integers summing to total.Each such list is equally likely to occur.
generate_Users - Returns customer data set according to the input parameters
CustomersGenerator - Distributes customers into nodes. Please note that for assuring the no-bottleneck assumption easily I put the big customers in the first node (you may need to change this)



P.S. sorry for the very ugly code
	
