__author__ = 'Majid Khonji'
import logging
from scipy import stats
import numpy as np
import networkx as nx


try:
    import gurobipy as gbp
except ImportError:
    logging.warning("Grubi not available!!")


def gurobi_setting(m):
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 200)
    m.setParam("MIPGapAbs", 0.00001)
    m.setParam("IntFeasTol",0.00001)
    m.setParam("DualReductions", 0) # without this, sometimes it returns error 4 (infeasible or unbounded)
    # m.setParam("MIPGapAbs", 0)
    # m.setParam("MIPGap", 0)
    # m.setParam("SolutionLimit", 1)
    m.setParam("NumericFocus",0) #0 automatic, 1-3 means how hard gurobi check numeric accuracty

def gurobi_handle_errors(m,algname=''):
    if m.status != gbp.GRB.status.OPTIMAL:
        # print("\t!!!!!!! Gurobi returned a non-OPT solution: status %d!!!!!!!!"%m.status)
        if (m.status == 3):
            logging.warning(" %s: Infeasible!"%algname)
            return False
        elif (m.status == 9):
            logging.warning("  %s: Time out!"%algname)
        elif (m.status == 13):
            logging.warning(" %s: SUBOPTIMAL 13	Unable to satisfy optimality tolerances; a sub-optimal solution is available."%algname)
        elif m.status == 12:
            logging.warning(" %s: NUMERIC: Optimization was terminated due to unrecoverable numerical difficulties."%algname)
            return False

        else:
            logging.warning(' %s: Failure: status %d'%(algname,m.status))
            return False
    return True

    #    print("\tstatus: %d" % m.status)


def print_instance(ins, detailed=True):
    if detailed: print "===== I N S T A N C E  D I S C ====="
    if detailed:
        print "-- : general"
        print " rounding_tolerance: ", ins.rounding_tolerance
        print " gen_cost: ", ins.gen_cost

    if detailed: print "-- network: "
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

def exactness_gap(ins, sol):
    T = ins.topology
    gap = {}
    for r in sol.l.keys():
        gap[r] = sol.l[r].X - sol.P[r].X**2 - sol.Q[r].X**2
    return gap


# sol.P - (P calculated from x)
def validate_capacity(ins,sol, output='gap_c'):
    P = {}
    x_p = {}
    gap_p = {}
    loss_p = {}
    Q = {}
    x_q = {}
    gap_q = {}
    loss_q = {}

    gap_c= {}

    T = ins.topology
    for e in ins.topology.edges():
        subtree_edges = nx.bfs_edges(T,e[1])
        z = T[e[0]][e[1]]['z']

        x_p[e] = np.sum([sol.x[k] * ins.loads_P[k] for k in ins.topology.edge[e[0]][e[1]]['K']])
        loss_p[e]= sol.l[e].x*z[0] + np.sum([sol.l[h].x*T[h[0]][h[1]]['z'][0] for h in subtree_edges])
        P[e] = x_p[e] + loss_p[e]
        gap_p[e] = sol.P[e].x - P[e]

        x_q[e] = np.sum([sol.x[k] * ins.loads_Q[k] for k in ins.topology.edge[e[0]][e[1]]['K']])
        loss_q[e]= sol.l[e].x*z[1] + np.sum([sol.l[h].x*T[h[0]][h[1]]['z'][1] for h in subtree_edges])
        Q[e] = x_q[e] + loss_q[e]
        gap_q[e] = sol.Q[e].x - Q[e]

        gap_c[e] = np.sqrt(T[e[0]][e[1]]['C']**2 - (gap_p[e]**2 + gap_q[e]**2 ))

    if output == 'gap_c':
        return gap_c
    elif output == 'gap_p':
        return gap_p
    elif output == 'P':
        return P
    elif output == 'x_p':
        return x_p
    elif output == 'l_p':
        return loss_p
    elif output == 'gap_q':
        return gap_q
    elif output == 'Q':
        return Q
    elif output == 'x_q':
        return x_q
    elif output == 'l_q':
        return loss_q

def mean_yerr(a, c=0.95):
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    yerr = se * stats.t.ppf((1 + c) / 2., n - 1)
    return m, yerr
