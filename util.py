__author__ = 'Majid Khonji'
import logging
from scipy import stats
import numpy as np
import networkx as nx


try:
    import gurobipy as gbp
except ImportError:
    logging.warning("Grubi not available!!")


def gurobi_setting(m, epsilon=None):
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 200)
    if epsilon != None:
        m.setParam("MIPGap", epsilon/(1-epsilon))
        # m.setParam("MIPGap", epsilon)
    # m.setParam("MIPGapAbs", 0.00001)
    # m.setParam("IntFeasTol",0.00001)
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
    # print(' C_: %s' % {e: round(ins.topology[e[0]][e[1]]['C_'], 3) for e in ins.topology.edges()})
    print(
        ' z : %s' % {e: (round(ins.topology[e[0]][e[1]]['z'][0], 3), round(ins.topology[e[0]][e[1]]['z'][1], 3)) for e
                     in
                     ins.topology.edges()})
    print ' V_ = ', (ins.v_0 - ins.v_min) / 2
    print "===================================="


def print_customer(ins, k):
    print('------- customer: %d -------' % k)
    print('demand: (%f,%f)' % (ins.loads_P[k], ins.loads_Q[k]))
    print 'Q values: ', {l: ins.Q[(k,l)] for l in ins.leaf_nodes}
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

    gap_l_p = {}
    gap_l_q = {}

    gap_c= {}

    T = ins.topology
    for e in ins.topology.edges():
        subtree_edges = nx.bfs_edges(T,e[1])
        z = T[e[0]][e[1]]['z']

        x_p[e] = np.sum([sol.x[k] * ins.loads_P[k] for k in ins.topology.edge[e[0]][e[1]]['K']])
        loss_p[e]= sol.l[e].x*z[0] + np.sum([sol.l[h].x*T[h[0]][h[1]]['z'][0] for h in subtree_edges])
        P[e] = x_p[e] + loss_p[e]
        gap_p[e] = sol.P[e].x - P[e]
        # gap_l_p[e] = sol.l[e].x - loss_p[e]/z[0]

        x_q[e] = np.sum([sol.x[k] * ins.loads_Q[k] for k in ins.topology.edge[e[0]][e[1]]['K']])
        loss_q[e]= sol.l[e].x*z[1] + np.sum([sol.l[h].x*T[h[0]][h[1]]['z'][1] for h in subtree_edges])
        Q[e] = x_q[e] + loss_q[e]
        gap_q[e] = sol.Q[e].x - Q[e]
        # gap_l_q[e] = sol.l[e].x - loss_q[e]/z[1]

        gap_c[e] = np.sqrt(T[e[0]][e[1]]['C']**2 - (P[e]**2 + Q[e]**2 ))
        # gap_c[e] = np.sqrt(T[e[0]][e[1]]['C']**2 - (sol.P[e].x**2 + sol.Q[e].x**2 ))

    if output == 'gap_c':
        return gap_c
    elif output == 'gap_p':
        return gap_p
    elif output == 'gap_l_p':
        return gap_l_p
    elif output == 'gap_l_q':
        return gap_l_q
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


def obj_min_loss_penalty(ins, sol, output='obj'):
    T = ins.topology
    loss = np.sum([sol.l[e].x*(T[e[0]][e[1]]['z'][0]) for e in T.edges()])
    penalty = 0
    for k in range(ins.n):
        penalty+= (1 - sol.x[k]) * ins.loads_utilities[k]
    if output=='obj':
        return penalty+loss
    if output=='obj2':
        return penalty+loss*ins.topology.graph['S_base']
    if output=='loss':
        return loss
    if output =='penalty':
        return penalty
def mean_yerr(a, c=0.950):
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    yerr = se * stats.t.ppf((1 + c) / 2., n - 1)
    return m, yerr

def check_ev_scheduling_fixed_int_sol_feasibility(ins,sol, fractional_sol=False):
    # testing capacity correcgtness
    total_ev_charge_at_time = {}
    for t in np.arange(ins.scheduling_horizon):
        total_ev_charge_at_time[t] = np.sum([ins.charging_rates[c] * sol.x[k, c]
                                             for (k,c) in ins.customers_at_time[t]] )
        # print total_ev_charge_at_time[t] + ins.base_load_over_time[t]
        assert (total_ev_charge_at_time[t] -ins.rounding_tolerance <= (ins.capacity_over_time[t] - ins.base_load_over_time[t])), "capacity violated at time %d"%t

    # testing y correctness
    energy_usage = {}
    for k in range(ins.n):
        energy_usage[k] = 0
        sum_of_x = 0
        for c in ins.customer_charging_options[k]:
            sum_of_x += sol.x[(k,c)]
            for t in ins.customer_charging_time_path[k,c]:
                energy_usage[k] += sol.x[(k, c)] * ins.charging_rates[c] * ins.step_length
                if not fractional_sol:
                    assert sol.x[(k,c)] in [1,0], "x[%d,%d] is fractional"%(k,c)
            assert sum_of_x <= 1, "sum X > 1! %f"%sum_of_x
        if not fractional_sol:
            assert (energy_usage[k] <= ins.customer_usage[k] and sum_of_x== 0) or (energy_usage[k] >= ins.customer_usage[k] and sum_of_x== 1), "x_c inconsistant"
            # print "satisfy ratio %d:"%k, energy_usage[k]/ins.customer_usage[k]

    print("solution is feasible!")
    return total_ev_charge_at_time
def check_ev_scheduling_sol_feasibility(ins,sol, fractional_sol=False):
    # testing capacity correcgtness
    total_ev_charge_at_time = {}
    for t in np.arange(ins.scheduling_horizon):
        total_ev_charge_at_time[t] = np.sum([ins.charging_rates[c] * sol.x[(k, c, t)]
                                                          for k in ins.customers_at_time[t] for c in
                                                          ins.customer_charging_options[k]])
        # print total_ev_charge_at_time[t] + ins.base_load_over_time[t]
        assert (total_ev_charge_at_time[t] -ins.rounding_tolerance <= (ins.capacity_over_time[t] - ins.base_load_over_time[t])), "capacity violated at time %d"%t

    # testing y correctness
    energy_usage = {}
    for k in range(ins.n):
        energy_usage[k] = 0
        if not fractional_sol:
            assert sol.y[k] in [0,1], "y is fractional"
        for t in ins.customer_charging_time_path[k]:
            sum_of_x = 0
            for c in ins.customer_charging_options[k]:
                energy_usage[k] += sol.x[(k, c, t)] * ins.charging_rates[c] * ins.step_length
                sum_of_x += sol.x[(k,c,t)]
                if not fractional_sol:
                    assert sol.x[(k,c,t)] in [1,0], "x[%d,%d,%d] is fractional"%(k,c,t)
            assert sum_of_x <= 1, "sum X > 1!"
        if not fractional_sol:
            assert (energy_usage[k] <= ins.customer_usage[k] and sol.y[k] == 0) or (energy_usage[k] >= ins.customer_usage[k] and sol.y[k] == 1), "y and x are inconsistant"
        # print "satisfy ratio %d:"%k, energy_usage[k]/ins.customer_usage[k]

    print("solution is feasible!")
    return total_ev_charge_at_time

def reduce_array_resolution(a, window=4):
    l = []
    tmp = 0
    for i, v in enumerate(a):
         if i % window != 0 or i == 0:
             tmp += v
         else:
             l.append(tmp/float(window))
             tmp = v
    l.append(tmp/float(window))
    return np.array(l)


# quick feasbility test using the linearized model
def check_opf_sol_feasibility(ins,x):

    T = ins.topology

    # testing capacity correctness
    for e in T.edges():
        rhs_P =  np.sum([x[k] * ins.loads_P[k] for k in np.intersect1d(T[e[0]][e[1]]['K'], x.keys())])
        rhs_Q =  np.sum([x[k] * ins.loads_Q[k] for k in np.intersect1d(T[e[0]][e[1]]['K'], x.keys())])


        if rhs_P**2 + rhs_Q**2 > T[e[0]][e[1]]['C'] ** 2:
            # print("solution is not feasible!")
            logging.warning("solution not feasible: Capacity violation")
            return False
    # testing voltage constraints
    for l in ins.leaf_nodes:
        C_V = np.sum([ins.Q[(k, l)] * x[k] for k in x.keys()])
        v_j = 1-2*C_V
        if ins.v_min > v_j:
            logging.warning("solution not feasible: Voltage violation")
            return False
    return True

