__author__ = 'majid'
import networkx as nx
import random
import numpy as np
import time
from maxOPF_instance import *

# cons = ["C" | "V" | ""] determines which constraint to consider
def greedy_card(ins, cons='C'):
    t1 = time.time()
    sol = maxOPF_sol()
    T = ins.topology
    order = np.argsort(ins.loads_S)
    sum_P = {e:0 for e in ins.topology.edges()}
    sum_Q = {e:0 for e in ins.topology.edges()}
    sum_V = {l:0 for l in ins.leaf_nodes}
    for k in order:
        # init
        tmp_sum_P = sum_P.copy()
        tmp_sum_Q = sum_Q.copy()
        tmp_sum_V = sum_V.copy()
        condition_C = False
        condition_V = False

        for l in ins.leaf_nodes:
            condition_V = sum_V[l] + ins.Q[(k,l)] <= (ins.v_0 - ins.v_min)/2
            if condition_V: tmp_sum_V[l] += ins.Q[(k,l)]
            else: break

        for e in ins.customer_path[k]:
            print 'edge', e, ' has capacity ', ins.topology[e[0]][e[1]]['C']
            condition_C = (sum_P[e] + ins.loads_P[k]) ** 2 + (sum_Q[e] + ins.loads_Q[k]) ** 2 <= ins.topology[e[0]][e[1]]['C'] ** 2
            if condition_C:
                tmp_sum_P[e] += ins.loads_P[k]
                tmp_sum_Q[e] += ins.loads_Q[k]
            else: break
        if cons == 'V': condition_C = True
        elif cons == 'C': condition_V = True

        if condition_C and condition_V:
            sum_P = tmp_sum_P.copy()
            sum_Q = tmp_sum_Q.copy()
            sum_V = tmp_sum_V.copy()
            sol.idx += [k]
            sol.obj += 1 #ins.loads_utilities[k]

    sol.running_time = time.time() - t1
    return sol


if __name__ == "__main__":
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    ins = rnd_instance()
    sol = greedy_card(ins)
    print 'obj = ', sol.obj

    # plt.show()
