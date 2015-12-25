__author__ = 'mkhonji'
import logging
from scipy import stats
import numpy as np


try:
    import gurobipy as gbp
except ImportError:
    logging.warning("Grubi not available!!")


def handle_errors(m):
    if m.status != gbp.GRB.status.OPTIMAL:
        # print("\t!!!!!!! Gurobi returned a non-OPT solution !!!!!!!!")
        if (m.status == 3):
            logging.warning("infeasible!")
            return False
        if (m.status == 9):
            logging.warning(' Time out!')
        elif (m.status == 13):
            logging.warning(' Suboptimal solution!')
        elif (m.status != 13):  # not even suboptimal
            logging.warning(' Failure')
            return False
        return True

        print("\tstatus: %d" % m.status)


def print_instance(ins, detailed=True):
    if detailed: print "===== I N S T A N C E  D I S C ====="
    if detailed: print "-- netork: "
    if detailed: print ' # nodes: ', len(ins.topology.nodes())
    if detailed: print ' # edges: ', len(ins.topology.edges())
    # print ' Edges:', ins.topology.edges()
    if detailed: print ' Leaf nodes:', ins.leaf_nodes
    print "-- customers:"
    print " I: ", ins.I
    print " F: ", ins.F
    print(' util:       %s' % str({k: round(ins.loads_utilities[k], 2) for k in
                                   range(ins.n)}))  # str(map(lambda x: round(x, 2), ins.loads_utilities)))
    print(' loads_S:    %s' % str(
        {k: round(ins.loads_S[k], 3) for k in range(ins.n)}))  # str(map(lambda x: round(x, 2), ins.loads_S)))
    print(' loads_P:    %s' % str(
        {k: round(ins.loads_P[k], 3) for k in range(ins.n)}))  # str(map(lambda x: round(x, 2), ins.loads_S)))
    print(' loads_Q:    %s' % str(
        {k: round(ins.loads_Q[k], 3) for k in range(ins.n)}))  # str(map(lambda x: round(x, 2), ins.loads_S)))
    print(' Q(k,l):     %s' % str({i[0]: round(i[1], 3) for i in ins.Q.iteritems()}))
    print "-- cons: "
    print(' C : %s' % {e: round(ins.topology[e[0]][e[1]]['C'], 3) for e in ins.topology.edges()})
    print(' C_: %s' % {e: round(ins.topology[e[0]][e[1]]['C_'], 3) for e in ins.topology.edges()})
    print(
        ' z : %s' % {e: (round(ins.topology[e[0]][e[1]]['z'][0], 3), round(ins.topology[e[0]][e[1]]['z'][1], 3)) for e
                     in
                     ins.topology.edges()})
    print ' V_ = ', (ins.v_0 - ins.v_min) / 2
    print "===================================="


def print_customer(ins, k):
    print('------- customer: %d -------' % k)
    print('demand: (%f,%f)' % (ins.loads_P[k], ins.loads_Q[k]))
    print 'Q values: ', {l: ins.Q[k:l] for l in ins.leaf_nodes}
    print('attached bus: %d' % ins.customer_node[k])
    print('path: %s' % str(ins.customer_path_nodes[k]))


def mean_yerr(a, c=0.95):
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    yerr = se * stats.t.ppf((1 + c) / 2., n - 1)
    return m, yerr
