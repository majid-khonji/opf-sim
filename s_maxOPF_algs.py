#!/usr/bin/python
__author__ = 'majid'

import networkx as nx
import numpy as np
import time
import instance as a
import logging, copy
import util as u

import OPF_algs as o

try: import gurobipy as gbp
except ImportError: logging.warning("Grubi not available!!")


# cons = ["C" | "V" | ""] determines which constraint to consider
# capacity_flag = ['C_'|'C'] tell which edge attribute in ins.topology corresponds to capacity
def greedy_card(ins, cons='', fixed_demands_P=None, fixed_demands_Q=None, capacity_flag='C_'):
    t1 = time.time()
    sol = a.maxOPF_sol()
    T = ins.topology
    idx = None
    if ins.I== None:
        idx = np.arange(ins.n)
    else:
        idx = ins.I
    # logging.info("         given idx: %s" % str(idx))
    order = np.argsort(np.array(ins.loads_S[idx]))
    sum_P = fixed_demands_P
    sum_Q = fixed_demands_Q
    if fixed_demands_P == None: sum_P = {e: 0 for e in ins.topology.edges()}
    if fixed_demands_Q == None: sum_Q = {e: 0 for e in ins.topology.edges()}
    sum_V = {l: 0 for l in ins.leaf_nodes}

    # print 'upper bound V = ', (ins.v_0 - ins.v_min)/2
    # logging.info("         order: %s"% str(idx[order]))
    for k in idx[[order]]:
        # print '------ customer %d -----'%k
        # print 'demand = (%f,%f)' % (ins.loads_P[k], ins.loads_Q[k])
        # init
        tmp_sum_P = sum_P.copy()
        tmp_sum_Q = sum_Q.copy()
        tmp_sum_V = sum_V.copy()
        condition_C = False
        condition_V = False

        for l in ins.leaf_nodes:
            # print 'Q_%d(%d) = %f'%(k,l, ins.Q[(k,l)])
            condition_V = sum_V[l] + ins.Q[(k, l)] <= (ins.v_0 - ins.v_min) / 2
            if condition_V:
                tmp_sum_V[l] += ins.Q[(k, l)]
            else:
                # print 'V violated with customer %d'%k
                break

        for e in ins.customer_path[k]:
            condition_C = (sum_P[e] + ins.loads_P[k]) ** 2 + (sum_Q[e] + ins.loads_Q[k]) ** 2 <= \
                          ins.topology[e[0]][e[1]][capacity_flag] ** 2
            if condition_C:
                tmp_sum_P[e] += ins.loads_P[k]
                tmp_sum_Q[e] += ins.loads_Q[k]
            else:
                # print 'C violated with customer %d'%k
                break
                # logging.info("         [%d,%s] Condition C :  %r"%(k, str(e), condition_C))
        if cons == 'V':
            condition_C = True
        elif cons == 'C':
            condition_V = True

        if condition_C and condition_V:
            sum_P = tmp_sum_P.copy()
            sum_Q = tmp_sum_Q.copy()
            sum_V = tmp_sum_V.copy()
            sol.idx += [k]
            sol.obj += 1  # ins.loads_utilities[k]

    sol.running_time = time.time() - t1
    return sol
def check_feasibility(ins, x, capacity_flag='C_'):
    T = ins.topology
    for e in T.edges():
        pure_demand_P = np.sum([ins.loads_P[k] * x[k] for k in T[e[0]][e[1]]['K']])
        pure_demand_Q = np.sum([ins.loads_Q[k] * x[k] for k in T[e[0]][e[1]]['K']])
        S = np.sqrt(pure_demand_P**2 + pure_demand_Q**2)
        C = T[e[0]][e[1]][capacity_flag]
        if S > C:
            print 'S = %.3f, C = %.3f'%(S,C)
            return False
    return True

# cons = ["C" | "V" | ""] determines which constraint to consider
# capacity_flag = ['C_'|'C'] tell which edge attribute in ins.topology corresponds to capacity
def greedy(ins, cons='', capacity_flag='C_', fixed_demands_P=None, fixed_demands_Q=None):
    ins_g = copy.copy(ins)
    t1 = time.time()
    L = np.max(ins.loads_utilities) / (ins.n ** 2 * 1.)
    rounded_util = np.floor(ins.loads_utilities/L)

    # logging.info('rounded util: %s' % str(rounded_util))
    I = ins.I.copy()
    # print 'initial I', I
    N = {}

    set_0 = np.argwhere((0 <= rounded_util)*(rounded_util <= 1)).ravel()
    set_0 = np.intersect1d(set_0, ins.I)
    if set_0.size != 0:
        N[0] = set_0
        I = np.setdiff1d(I, N[0])
    i = 0
    while I.size != 0:
        i += 1
        set_i = np.argwhere((2**i <= rounded_util)* (rounded_util < 2**(i+1))).ravel()
        set_i = np.intersect1d(set_i,ins.I)
        if set_i.size != 0:
            N[i] = set_i
            I = np.setdiff1d(I, N[i])
            # print 'N[%d]= '%i, N[i]
            # print 'util = ', [rounded_util[k] for k in N[i]]
    groups = N.keys()
    max_sol = a.maxOPF_sol()
    max_sol.obj = 0
    for i in groups:
        ins_g.I = N[i]
        # logging.info('group(%d)' % i)
        sol = greedy_card(ins_g, cons, fixed_demands_P=fixed_demands_P, fixed_demands_Q=fixed_demands_Q,
                          capacity_flag=capacity_flag)
        obj_val = np.sum(ins_g.loads_utilities[sol.idx])
        # logging.info('         obj val = %f' % (obj_val))
        # logging.info('         sol idx = %s' % str(sol.idx))
        if obj_val > max_sol.obj:
            max_sol.obj = obj_val
            max_sol.idx = sol.idx

    max_sol.x = {k: 0 for k in ins.I}
    for k in max_sol.idx: max_sol.x[k] = 1
    max_sol.groups = groups
    max_sol.running_time = time.time() - t1
    # print 'actual obj = ', max_sol.obj
    # print 'idx', max_sol.idx
    # print 'I', ins.I
    # print '  sure?    = ', sum([ins.loads_utilities[k]*max_sol.x[k] for k in ins.I])
    return max_sol

# cons = ["C" | "V" | ""] determines which constraint to consider
# capacity_flag = ['C_'|'C'] tell which edge attribute in ins.topology corresponds to capacity
def greedy_slow(ins, cons='', capacity_flag='C_', fixed_demands_P=None, fixed_demands_Q=None):
    ins_g = copy.copy(ins)
    t1 = time.time()
    L = np.max(ins.loads_utilities) / (ins.n ** 2 * 1.)

    rounded_util = [np.floor(u / L) for u in ins.loads_utilities]
    # logging.info('rounded util: %s' % str(rounded_util))
    actual_time = time.time() - t1
    I = set(ins.I)
    # print 'initial I', I
    N = {}

    set_0 = {k for k in ins.I if 0 <= rounded_util[k] <= 1}
    if set_0 != set():
        N[0] = set_0
        I = I ^ N[0]
    i = 0
    while I != set():
        i += 1
        set_i = {k for k in I if 2 ** i <= rounded_util[k] < 2 ** (i + 1)}
        # print i, set_i
        if set_i != set():
            I = I ^ set_i
            N[i] = set_i
            # print 'N[%d]= '%i, N[i]
            # print 'util = ', [rounded_util[k] for k in N[i]]
    groups = N.keys()
    max_sol = a.maxOPF_sol()
    max_sol.obj = 0
    for i in groups:
        ins_g.I = np.array(list(N[i]))
        # logging.info('group(%d)' % i)
        sol = greedy_card(ins_g, cons, fixed_demands_P=fixed_demands_P, fixed_demands_Q=fixed_demands_Q,
                          capacity_flag=capacity_flag)
        obj_val = np.sum([ins_g.loads_utilities[k] for k in sol.idx])
        # logging.info('         obj val = %f' % (obj_val))
        # logging.info('         sol idx = %s' % str(sol.idx))
        if obj_val > max_sol.obj:
            max_sol.obj = obj_val
            max_sol.idx = sol.idx
    max_sol.x = {k: 0 for k in ins.I}
    for k in max_sol.idx: max_sol.x[k] = 1
    max_sol.groups = groups
    max_sol.running_time = time.time() - t1
    return max_sol

# cons = ["C" | "V" | ""] determines which constraint to consider
# capacity_flag = ['C_'|'C'] tell which edge attribute in ins.topology corresponds to capacity
def mixed_greedy(ins, cons='', capacity_flag='C_'):
    t1 = time.time()
    ins2 = copy.copy(ins)
    ins2.F = np.append(ins.F, ins.I)
    ins2.I = np.array([])
    sol_f = o.max_OPF_OPT(ins2, cons)
    # ins2.I = np.array([k for k,v in sol_f.x.iteritems() if (1-epsilon<= sol_f.x[k]<= 1+epsilon)  and k in ins.I])
    ins2.I = ins.I  # np.array([k for k,v in sol_f.x.iteritems() if ((1-epsilon<= sol_f.x[k]<= 1+epsilon) or (-epsilon<= sol_f.x[k]<= epsilon)) and k in ins.I])
    ins2.F = np.array([])
    T = ins2.topology
    fixed_demands_P = {e: 0 for e in T.edges()}
    fixed_demands_Q = {e: 0 for e in T.edges()}
    for e in T.edges():
        # subtract demands passing through e from its capacity
        index_set = set(T[e[0]][e[1]]['K']).intersection(set(ins.F))
        # print index_set
        fixed_demands_P[e] = np.sum([ins2.loads_P[k] * sol_f.x[k] for k in index_set])
        fixed_demands_Q[e] = np.sum([ins2.loads_Q[k] * sol_f.x[k] for k in index_set])
        # T[e[0]][e[1]]['C_'] = T[e[0]][e[1]]['C_'] - np.sqrt(fixed_demands_P**2 + fixed_demands_Q**2)

    sol_g = greedy(ins2, cons=cons, fixed_demands_P=fixed_demands_P, fixed_demands_Q=fixed_demands_Q,
                   capacity_flag='C_')

    # print 'x ', sol_g.x
    fraction_obj = np.sum(ins.loads_utilities[k] * sol_f.x[k] for k in ins.F)
    # print 'actual obj = ', sol_g.obj
    # print '  sure?    = ', sum([ins2.loads_utilities[k]*sol_g.x[k] for k in ins.I])
    # print 'added _obj = ', fraction_obj
    fraction_x = {k: sol_f.x[k] for k in ins.F}
    sol_g.x.update(fraction_x)
    # print 'fraction ', fraction_x
    # print 'greedy full sol = ', sol_g.x

    sol_g.obj += fraction_obj
    sol_g.running_time = time.time() - t1

    return sol_g
# estimate loss through small steps
def adaptive_greedy(ins,  cons='', loss_step = .005):
   
    ins2 = copy.copy(ins)
    T = ins2.topology
    loss_ratio = 0
    for e in T.edges(): T[e[0]][e[1]]['C_'] = T[e[0]][e[1]]['C']
    sol = None
    if ins.F.size == 0: sol = greedy(ins,cons,capacity_flag='C_')
    else: sol = mixed_greedy(ins2,cons,capacity_flag='C_')
    _sol_ = o.min_loss_OPF(ins2,sol.x)
    attemps = 0
    while _sol_.obj == -np.infty:
        attemps+=1
        loss_ratio += loss_step
        for e in T.edges(): T[e[0]][e[1]]['C_'] = T[e[0]][e[1]]['C']*(1-loss_ratio)
        if ins.F.size == 0: sol = greedy(ins,cons,capacity_flag='C_')
        else: sol = mixed_greedy(ins2,cons,capacity_flag='C_')
        _sol_ = o.min_loss_OPF(ins2,sol.x)
        # print attemps,
    sol.loss_ratio = loss_ratio

    return sol

# estimate loss through small steps
def adaptive_OPT(ins,  cons='', loss_step = .005):
    ins2 = copy.copy(ins)
    T = ins2.topology
    loss_ratio = 0
    for e in T.edges(): T[e[0]][e[1]]['C_'] = T[e[0]][e[1]]['C']
    sol = OPT(ins2,cons,capacity_flag='C_')
    _sol_ = o.min_loss_OPF(ins2,sol.x)
    attemps = 0
    while _sol_.obj == -np.infty:
        attemps+=1
        loss_ratio += loss_step
        for e in T.edges(): T[e[0]][e[1]]['C_'] = T[e[0]][e[1]]['C']*(1-loss_ratio)
        sol = OPT(ins2,cons,capacity_flag='C_')
        _sol_ = o.min_loss_OPF(ins2,sol.x)
        # print attemps,
    sol.loss_ratio = loss_ratio

    return sol




# cons = ["C" | "V" | ""] determines which constraint to consider
# capacity_flag = ['C_'|'C'] tell which edge attribute in ins.topology corresponds to capacity
def OPT_slow(ins, cons='', capacity_flag='C_'):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)

    x = {k: 0 for k in range(ins.n)}
    for i in ins.I: x[i] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % i)
    for i in ins.F: x[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % i)
    m.update()

    obj = gbp.quicksum(x[k] * ins.loads_utilities[k] for k in range(ins.n))
    m.setObjective(obj, gbp.GRB.MAXIMIZE)

    if cons == 'C' or cons == '':
        # capacity constraints
        for e in T.edges():
            if T[e[0]][e[1]]['K'] != []:
                capacity_cons = gbp.quicksum([x[i] * ins.loads_P[i] for i in T[e[0]][e[1]]['K']]) * \
                                gbp.quicksum([x[i] * ins.loads_P[i] for i in T[e[0]][e[1]]['K']]) + \
                             gbp.quicksum([x[i] * ins.loads_Q[i] for i in T[e[0]][e[1]]['K']]) * \
                             gbp.quicksum([x[i] * ins.loads_Q[i] for i in T[e[0]][e[1]]['K']])
                m.addQConstr(capacity_cons <= T[e[0]][e[1]][capacity_flag] ** 2, "C%s" % str(e))
    if cons == 'V' or cons == '':
        V = (ins.v_0 - ins.v_min)/2.
        for l in ins.leaf_nodes:
            voltage_cons = gbp.quicksum([x[k] * ins.Q[(k,l)] for k in range(ins.n)])
            m.addConstr(voltage_cons <= V)
    for i in ins.F:
        m.addConstr(x[i]<=1, "x[%d]: ub")
        m.addConstr(x[i]>=0, "x[%d]: lb")

    m.update()
    m.optimize()

    sol = a.maxOPF_sol()
    sol.status = m.status
    sol.running_time = time.time() - t1
    if (u.gurobi_handle_errors(m) == False):
        sol.obj = -np.inf
        return sol
    sol.obj = obj.getValue()
    sol.x = {k: x[k].x for k in range(ins.n)}
    sol.idx = [i for i, x_i in enumerate(m.getVars()) if x_i.x == 1]

    # print 'actual solution x: ', [x_i for x_i in enumerate(m.getVars())]
    sol.running_time = time.time() - t1

    return sol


# more efficient formulation
# cons = ["C" | "V" | ""] determines which constraint to consider
# capacity_flag = ['C_'|'C'] tell which edge attribute in ins.topology corresponds to capacity
def OPT(ins, cons='', capacity_flag='C_'):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)
    x = {k: 0 for k in range(ins.n)}
    P = {e: 0 for e in T.edges()}
    Q = {e: 0 for e in T.edges()}
    v = {i: 0 for i in T.nodes()}
    v[0] = ins.v_0
    dummy_p = {i: 0 for i in T.nodes()}
    dummy_q = {i: 0 for i in T.nodes()}
    for k in T.nodes(): dummy_p[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_p[%d]" % k)
    for k in T.nodes(): dummy_q[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_q[%d]" % k)
    for i in ins.I: x[i] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % i)
    for i in ins.F: x[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % i)
    for e in T.edges(): P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
    for e in T.edges(): Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    for k in T.nodes()[1:]:
        v[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % k)
    m.update()

    obj = gbp.quicksum(x[k] * ins.loads_utilities[k] for k in range(ins.n))
    m.setObjective(obj, gbp.GRB.MAXIMIZE)

    for e in T.edges():
        m.addConstr(dummy_p[e[1]] >= 0, "dummy_P_%d" % e[1])
        rhs_P = gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [P[(e[1], h)] for h in T.edge[e[1]].keys() if e[1] < h]) + dummy_p[e[1]]
        rhs_Q = gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys() if e[1] < h]) + dummy_q[e[1]]
        m.addQConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s=" % str(e))
        m.addQConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))

        if cons == 'C' or cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]][capacity_flag] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if cons == 'V' or cons == '':
            z = T[e[0]][e[1]]['z']
            rhs_v = v[e[0]] - 2 * (z[0] * P[e] + z[1] * Q[e])
            m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d=" % e[1])
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d" % e[1])  # voltage constraint
    for i in ins.F:
        m.addConstr(x[i] <= 1, "x[%d]: ub")
        m.addConstr(x[i] >= 0, "x[%d]: lb")

    m.update()
    m.optimize()

    sol = a.maxOPF_sol()
    sol.status = m.status
    sol.running_time = time.time() - t1
    if (u.gurobi_handle_errors(m) == False):
        sol.obj = -np.inf
        return sol
    sol.obj = obj.getValue()
    sol.x = {k: x[k].x for k in range(ins.n)}
    sol.idx = [i for i, x_i in enumerate(m.getVars()) if x_i.x == 1]

    # print 'actual solution x: ', [x_i for x_i in enumerate(m.getVars())]
    sol.running_time = time.time() - t1

    return sol



# def handleError(self, record):
#     raise
if __name__ == "__main__":
    import sys
    # logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    # root = logging.getLogger()
    # root.setLevel(logging.INFO)
    # ch = logging.StreamHandler(sys.stdout)
    # formatter = logging.Formatter('%(levelname)s: %(message)s')
    # ch.setFormatter(formatter)
    # root.addHandler(ch)


    # ins = a.rnd_tree_instance(n=1000, depth=10, branch=2, capacity_range=(50, 50), util_func=lambda x,y: x-x + x**2)
    # ins = a.rnd_path_instance(n=100,node_count=10, capacity_range=(30, 50))
    T = a.network_38node(loss_ratio=.05)
    ins = a.sim_instance(T, scenario='FUR', n = 1000,F_percentage=.0)
    # ins = a.rnd_instance_from_graph(T, n=1000)
    # ins = a.sim_instance(T, scenario='FCM', n = 100)
    # ins = a.single_link_instance(capacity=10,n=100)
    # ins.loads_utilities = np.ones(ins.n)
    # ins.loads_Q = np.zeros(ins.n)
    # ins.loads_S = ins.loads_P
    # ins.F = ins.I[80:100]
    # ins.I = ins.I[0:80]
    #    u.print_instance(ins)

    # print '----- OPT ------'
    # sol2 = OPT_slow(ins, cons='C', capacity_flag='C')
    # print 'slow, obj value: ', sol2.obj
    # print sol2.x.values()
    # print check_feasibility(ins,sol2.x)
    sol2 = OPT(ins, cons='C', capacity_flag='C_')
    print 'OPT, obj value     : ', sol2.obj

    sol2 = adaptive_OPT(ins, cons='C')
    print 'adaptive, obj value: ', sol2.obj
    print 'loss ratio ', sol2.loss_ratio

    # print sol2.x.values()
    # print check_feasibility(ins,sol2.x)
    # print 'loss = ', sol2.loss_ratio
    # print 'obj x: ', sol2.x
    # print 'sol x = %s' % str(map(lambda a: round(a, 2), sol2.x))
    # print 'time: ', sol2.running_time

    # print '----- mixed greedy ------'
    # sol_f = mixed_greedy(ins, cons='C', capacity_flag='C_')
    # sol_f.ar = sol_f.obj / sol2.obj
    # print "max value =", sol_f.obj
    # print 'again     =', sum([ins.loads_utilities[k]*sol_f.x[k] for k in range(ins.n)])
    # print 'max sol x = %s' % str(map(lambda a: round(a, 2), sol_f.x))
    # print 'len x', len(sol_f.x)
    # print '# groups: ', len(sol_f.groups)
    # print 'time: ', sol_f.running_time
    # print '----- greedy slow ------'
    # t1 = time.time()
    # sol_f = greedy_slow(ins, cons='C', capacity_flag='C')
    # print 'in %f sec'% (time.time() - t1)
    #    sol_f.ar = sol_f.obj/sol2.obj
    # print "max value =",  sol_f.obj

    # print '----- greedy fast ------'
    # t1 = time.time()
    # sol_f = greedy(ins, cons='C', capacity_flag='C')

    # print 'in %f sec'% (time.time() - t1)
    #    sol_f.ar = sol_f.obj/sol2.obj
    # print "max value =",  sol_f.obj
    #    print 'max sol idx = %s' % str(sol_f.idx)
    #    print 'max sol idx = %s' % str(sol_f.idx)
    #    print '# groups: ', len(sol_f.groups)
    #    print 'time: ', sol_f.running_time

    # print '\n=== ratio %.3f ===' % sol_f.ar




    # sol = greedy_card(ins)
    # print 'obj = ', sol.obj
    # print 'idx = ', sol.idx
    # T = ins.topology
    # pos = nx.spring_layout(T)
    # nx.draw(T, pos)
    # node_labels = nx.get_node_attributes(T,'customers')
    # nx.draw_networkx_labels(T, pos, labels = node_labels)
    # edge_labels = nx.get_edge_attributes(T,'z')
    # nx.draw_networkx_edge_labels(T, pos, labels = edge_labels)
    # plt.show()
