#!/usr/bin/python
__author__ = 'Majid Khonji'
import numpy as np
import time
import instance as a
import logging
import networkx as nx
import util as u
import itertools

try:
    import gurobipy as gbp
except ImportError:
    logging.warning("Grubi not available!!")


def PTAS(ins, guess_set_size=1, use_LP=True):
    best_sol = a.OPF_sol();
    best_sol.obj = np.inf

    sol = round_OPF(ins, use_LP=use_LP, guess_x={})
    if sol.succeed:
        print best_sol.obj
        if sol.obj < best_sol.obj and sol.succeed:
            best_sol = sol

    for size in np.arange(1, guess_set_size + 1):
        for guess_I1 in itertools.combinations(np.arange(ins.n), size):
            min_util = np.min([ins.loads_utilities[k] for k in guess_I1])
            guess_I2 = {k:0 for k in ins.I if ins.loads_utilities[k] <= min_util}
            guess_x = {k: 1 for k in guess_I1}
            for k in guess_I2:
                guess_x[k] = 0

            sol = round_OPF(ins, use_LP=use_LP, guess_x=guess_x)
            if sol.succeed:
                print best_sol.obj
                if sol.obj < best_sol.obj and sol.succeed:
                    best_sol = sol
    return best_sol


# we use the same objective as min_OPF_OPT: penalty + loss + 1
# some nasty tricks: if solve_remaining fails, we take loss from the fractional solution (its a bound, right?)
def round_OPF(ins, use_LP=True, guess_x={}, alg='min_OPF_round'):
    T = ins.topology
    t1 = time.time()
    sol = None
    if not ins.util_max_objective:
        sol = min_OPF_OPT(ins, guess_x=guess_x, fractional=True)
    elif ins.util_max_objective and ins.drop_l_terms:
        sol = max_sOPF_OPT(ins, guess_x=guess_x, fractional=True)
    sol.frac_comp_count = 0
    if sol.succeed:
        # print 'calling lp'
        customers = np.setdiff1d(ins.I, guess_x)
        sol_lp = sol
        if use_LP:
            sol_lp = _LP(ins, sol, customers)
        if sol_lp.succeed:
            for k in customers:
                sol.x[k] = sol_lp.x[k]

            for k in customers:
                if ins.rounding_tolerance < sol.x[k] < 1 - ins.rounding_tolerance:
                    sol.x[k] = 0
                    sol.frac_comp_count += 1
            if ins.util_max_objective:
                obj = 0
                for k in range(ins.n):
                    obj += sol.x[k] * ins.loads_utilities[k]
                sol.obj = obj
                sol.frac_comp_percentage = sol.frac_comp_count/(ins.n*1.)*100
                sol.running_time = time.time() - t1
                return sol

            else:
                obj = 1
                for k in range(ins.n):
                    obj += (1 - sol.x[k]) * ins.loads_utilities[k]
                if not ins.drop_l_terms:
                    sol2 = _solve_remaining(ins, guess_x=sol.x)
                    if sol2.succeed:
                        obj += T.graph['S_base'] * sol2.obj
                    else:
                        obj += T.graph['S_base'] * u.obj_min_loss_penalty(ins,sol,output='loss')
                        sol2.x = sol.x
                        sol2.succeed = True
                    sol2.obj = obj
                    sol2.frac_comp_count = sol.frac_comp_count
                    sol2.frac_comp_percentage = sol2.frac_comp_count/(ins.n*1.)*100

                    sol2.running_time = time.time() - t1
                    return sol2
    sol.succeed = False
    return sol


def _LP(ins, sol, customers=[], alg='lp'):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)

    x = {}
    for k in customers: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)


    if ins.util_max_objective:
        obj = gbp.quicksum(x[k] * ins.loads_utilities[k] for k in customers)
        m.setObjective(obj, gbp.GRB.MAXIMIZE)
    else:
        obj = gbp.quicksum((1 - x[k]) * ins.loads_utilities[k] for k in customers)
        m.setObjective(obj, gbp.GRB.MINIMIZE)

    if ins.cons == '' or ins.cons == 'C':
        for e in T.edges():
            # print "edge ", e
            edge_customers = np.intersect1d(customers, T[e[0]][e[1]]['K'])

            C_P = np.sum([sol.x[k] * ins.loads_P[k] for k in edge_customers])
            # print C_P

            lhs_P = gbp.quicksum([x[k] * ins.loads_P[k] for k in edge_customers])
            m.addConstr(lhs_P, gbp.GRB.LESS_EQUAL, C_P , "Cp_%s" % str(e))

            C_Q = np.sum([sol.x[k] * ins.loads_Q[k] for k in edge_customers])
            lhs_Q = gbp.quicksum([x[k] * ins.loads_Q[k] for k in edge_customers])
            m.addConstr(lhs_Q, gbp.GRB.LESS_EQUAL, C_Q , "Cq_%s" % str(e))
    if not ins.drop_l_terms and (ins.cons == '' or ins.cons == 'V'):
        for l in ins.leaf_nodes:
            C_V = np.sum([ins.Q[(k, l)] * sol.x[k] for k in customers])
            lhs_V = gbp.quicksum([ins.Q[(k, l)] * x[k] for k in customers])
            m.addConstr(lhs_V, gbp.GRB.LESS_EQUAL, C_V , "Cv_%s" % str(l))
    for k in customers:
        m.addConstr(x[k] <= 1, "x[%d]: ub")
        # m.addConstr(x[k] >= 0, "x[%d]: lb")

    m.update()
    # m.computeIIS()
    # m.write("model.ilp")
    m.optimize()

    sol = a.OPF_sol()
    sol.running_time = time.time() - t1
    sol.gurobi_model = m

    if u.gurobi_handle_errors(m, algname=alg):
        sol.obj = obj.getValue()
        sol.x = {k: x[k].x for k in customers}
        # logging.info("\tx            = %s" % str(sol.x))
        # logging.info("\t{k: x_k>0}   = %s" % str([k for k in range(ins.n) if sol.x[k] > 0]))
        sol.succeed = True
    else:
        sol.succeed = False

    return sol

# minimizes loss
def _solve_remaining(ins, guess_x={}, alg="solve_remaining"):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)
    x = guess_x
    v = {}
    v[0] = ins.v_0
    l = {}
    P = {}
    Q = {}

    for i in T.nodes()[1:]:
        v[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % i)
    for e in T.edges(): l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
    for e in T.edges(): P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
    for e in T.edges(): Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    # m.update()

    ########## Objective #######
    root_edge = (0, T.edge[0].keys()[0])
    # obj = gbp.quicksum((1 - x[k]) * ins.loads_utilities[k] for k in range(ins.n)) + (ins.gen_cost) * P[
    #     root_edge]
    # obj = ins.gen_cost * P[root_edge]
    obj = gbp.quicksum([ T[e[0]][e[1]]['z'][0]* l[e] for e in T.edges()])
    m.setObjective(obj, gbp.GRB.MINIMIZE)
    for e in T.edges():
        z = T[e[0]][e[1]]['z']
        subtree_edges = nx.bfs_edges(T,e[1])
        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i
        # rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [P[(e[1], h)] for h in T.edge[e[1]].keys() ])
        # rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [P[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])
        rhs_P = l[e] * z[0] + gbp.quicksum([x[k] * ins.loads_P[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h]*z[0] for h in subtree_edges])
        m.addConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s " % str(e))

        # rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [Q[(e[1], h)] for h in T.edge[e[1]].keys()])
        # rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [Q[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])
        rhs_Q = l[e] * z[1] + gbp.quicksum([x[k] * ins.loads_Q[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h]*z[1] for h in subtree_edges])
        m.addConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s " % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d " % e[1])
        # m.update()

        if ins.cons == 'C' or ins.cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if ins.cons == 'V' or ins.cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d bound" % e[1])  # voltage constraint

        m.addConstr(v[e[1]] >= 0, "v_%d+"%e[1])
        m.addConstr(l[e] >= 0, "l_%s+"%str(e))
    m.update()
    # m.computeIIS()
    # m.write("model.ilp")
    m.optimize()

    sol = a.OPF_sol()
    sol.running_time = time.time() - t1
    sol.gurobi_model = m

    if u.gurobi_handle_errors(m, algname=alg):
        sol.obj = obj.getValue()
        sol.x = {k: x[k] for k in range(ins.n)}
        sol.l = l
        sol.P = P
        sol.Q = Q
        sol.v = v
        first_node = T.edge[0].keys()[0]
        sol.P_0 = P[(0, first_node)].X
        logging.info("\tx            = %s" % str(sol.x))
        logging.info("\t{k: x_k>0}   = %s" % str([k for k in range(ins.n) if sol.x[k] > 0]))
        sol.succeed = True
    else:
        sol.succeed = False

    return sol

# for FnT 2017
# No l terms and no voltage
def max_sOPF_OPT(ins, guess_x={}, fractional=False, debug=False,tolerance=0.001, alg="min_OPF_OPT"):
    assert(ins.cons == '' or ins.cons == 'V' or ins.cons == 'C' )
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("max_sOPF_OPT")

    u.gurobi_setting(m)
    x = {}
    P = {}
    Q = {}
    if fractional:
        for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    else:
        for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % k)

    for k in ins.F: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    for e in T.edges():
        P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
        Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))


    obj = gbp.quicksum(x[k] * ins.loads_utilities[k] for k in range(ins.n))
    m.setObjective(obj, gbp.GRB.MAXIMIZE)

    for e in T.edges():
        rhs_P = gbp.quicksum([x[k] * ins.loads_P[k] for k in T[e[0]][e[1]]['K']])
        m.addConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s" % str(e))

        rhs_Q = gbp.quicksum([x[k] * ins.loads_Q[k] for k in T[e[0]][e[1]]['K']])

        m.addConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s" % str(e))



        m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint

    for i in ins.F:
        m.addConstr(x[i] <= 1, "x[%d]: ub")
        # m.addConstr(x[i] >= 0, "x[%d]: lb")
    if fractional:
        for i in ins.I:
            m.addConstr(x[i] <= 1, "x[%d]: ub")
            # m.addConstr(x[i] >= 0, "x[%d]: lb")
    for k in guess_x.keys():
        m.addConstr(x[k], gbp.GRB.EQUAL, guess_x[k], "x[%d]: guess " % k)

    m.update()
    #m.write('model.lp')
    m.optimize()

    sol = a.OPF_sol()
    sol.running_time = time.time() - t1
    sol.gurobi_model = m

    if u.gurobi_handle_errors(m, algname=alg):
        sol.x = {k: x[k].x for k in range(ins.n)}
        sol.obj = obj.getValue()
        sol.P = P
        sol.Q = Q
        first_node = T.edge[0].keys()[0]
        sol.P_0 = P[(0, first_node)].X
        sol.succeed = True
    else:
        sol.succeed = False

    return sol

# for TPS 2017
# it has some quick and dirty tricks to make l tight
def min_OPF_OPT(ins, guess_x={}, fractional=False, debug=False,tolerance=0.001, alg="min_OPF_OPT"):
    assert(ins.cons == '' or ins.cons == 'V' or ins.cons == 'C' )
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("min_OPF_OPT")

    u.gurobi_setting(m)
    x = {}
    v = {}
    v[0] = ins.v_0
    l = {}
    P = {}
    Q = {}
    # dummy_p = {}; dummy_q = {}
    if fractional:
        for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    else:
        for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % k)

    for k in ins.F: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    for i in T.nodes()[1:]:
        v[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % i)
        # dummy_p[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_p_%d" % i)
        # dummy_q[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_q_%d" % i)
    for e in T.edges():
        l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
        P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
        Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))

    ########## Objective #######
    # root_edge = (0, T.edge[0].keys()[0])
    # obj = gbp.quicksum((1 - x[k]) * ins.loads_utilities[k] for k in range(ins.n)) + (ins.gen_cost) * P[
    #     root_edge]

    # loss minimization
    obj = 1 + gbp.quicksum((1 - x[k]) * ins.loads_utilities[k] for k in range(ins.n)) +\
          T.graph['S_base'] * gbp.quicksum([l[e]*(T[e[0]][e[1]]['z'][0]) for e in T.edges()])
    m.setObjective(obj, gbp.GRB.MINIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']
        subtree_edges = nx.bfs_edges(T,e[1])

        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i

        # rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [P[(e[1], h)] for h in T.edge[e[1]].keys()])  # + dummy_p[e[1]]
        rhs_P = l[e] * z[0] + gbp.quicksum([x[k] * ins.loads_P[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h]*z[0] for h in subtree_edges])

        m.addConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s" % str(e))
        # m.addConstr(P[e], gbp.GRB.LESS_EQUAL, rhs_P+tolerance, "P_%s" % str(e))
        # m.addConstr(P[e], gbp.GRB.GREATER_EQUAL, rhs_P-tolerance, "P_%s" % str(e))

        # rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [Q[(e[1], h)] for h in T.edge[e[1]].keys()]) # + dummy_q[e[1]]
        rhs_Q = l[e] * z[1] + gbp.quicksum([x[k] * ins.loads_Q[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h]*z[1] for h in subtree_edges])

        m.addConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s" % str(e))
        # m.addConstr(Q[e], gbp.GRB.LESS_EQUAL, rhs_Q+tolerance, "Q_%s" % str(e))
        # m.addConstr(Q[e], gbp.GRB.GREATER_EQUAL, rhs_Q-tolerance, "Q_%=" % str(e))


        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d" % e[1])



        if ins.cons == 'C' or ins.cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if ins.cons == 'V' or ins.cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d" % e[1])  # voltage constraint

        m.addConstr(v[e[1]] >= 0, "v_%d+:"%e[1])
        m.addConstr(l[e] >= 0, "l_%s+:"%str(e))



        # m.addConstr(dummy_p[e[1]], gbp.GRB.GREATER_EQUAL, 0, "dummy_P_%d" % e[1])
        # m.addConstr(dummy_q[e[1]], gbp.GRB.GREATER_EQUAL, 0, "dummy_Q_%d" % e[1])

    for i in ins.F:
        m.addConstr(x[i] <= 1, "x[%d]: ub")
        # m.addConstr(x[i] >= 0, "x[%d]: lb")
    if fractional:
        for i in ins.I:
            m.addConstr(x[i] <= 1, "x[%d]: ub")
            # m.addConstr(x[i] >= 0, "x[%d]: lb")
    for k in guess_x.keys():
        m.addConstr(x[k], gbp.GRB.EQUAL, guess_x[k], "x[%d]: guess " % k)

    m.update()
    #m.write('model.lp')
    m.optimize()

    sol = a.OPF_sol()
    sol.running_time = time.time() - t1
    sol.gurobi_model = m

    if u.gurobi_handle_errors(m, algname=alg):
        sol.x = {k: x[k].x for k in range(ins.n)}
        sol.obj = obj.getValue()
        sol.l = l
        sol.P = P
        sol.Q = Q
        sol.v = v
        first_node = T.edge[0].keys()[0]
        sol.P_0 = P[(0, first_node)].X
        sol.succeed = True
    else:
        sol.succeed = False

    return sol

def min_OPF_OPT_erroneous(ins, guess_x={}, fractional=False, debug=False,tolerance=0.001, alg="min_OPF_OPT"):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)
    x = {}
    v = {}
    v[0] = ins.v_0
    l = {}
    P = {}
    Q = {}
    if fractional:
        for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    else:
        for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % k)

    for k in ins.F: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    for i in T.nodes()[1:]:
        v[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % i)
    for e in T.edges(): l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
    for e in T.edges(): P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
    for e in T.edges(): Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    # m.update()

    ########## Objective #######
    root_edge = (0, T.edge[0].keys()[0])
    obj = gbp.quicksum((1 - x[k]) * ins.loads_utilities[k] for k in range(ins.n)) + (ins.gen_cost) * P[
        root_edge]
    # obj = gbp.quicksum((1 - x[k]) * ins.loads_utilities[k] for k in range(ins.n)) + gbp.quicksum([l[e]*(T[e[0]][e[1]]['z'][0]) for e in T.edges()])
    m.setObjective(obj, gbp.GRB.MINIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']
        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i
        rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [P[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])  # + dummy_p[e[1]]
        # m.addConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s=" % str(e))
        m.addConstr(P[e], gbp.GRB.GREATER_EQUAL, rhs_P-tolerance, "P_%s=" % str(e))
        m.addConstr(P[e], gbp.GRB.LESS_EQUAL, rhs_P+tolerance, "P_%s=" % str(e))

        rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])  # + dummy_q[e[1]]
        # m.addConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))
        m.addConstr(Q[e], gbp.GRB.GREATER_EQUAL, rhs_Q-tolerance, "Q_%s=" % str(e))
        m.addConstr(Q[e], gbp.GRB.LESS_EQUAL, rhs_Q+tolerance, "Q_%s=" % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d=" % e[1])

        if ins.cons == 'C' or ins.cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if ins.cons == 'V' or ins.cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d" % e[1])  # voltage constraint
    for i in ins.F:
        m.addConstr(x[i] <= 1, "x[%d]: ub")
        m.addConstr(x[i] >= 0, "x[%d]: lb")
    if fractional:
        for i in ins.I:
            m.addConstr(x[i] <= 1, "x[%d]: ub")
            m.addConstr(x[i] >= 0, "x[%d]: lb")
    for k in guess_x.keys():
        m.addConstr(x[k], gbp.GRB.EQUAL, guess_x[k], "x[%d]: guess " % k)

    m.update()
    m.optimize()

    sol = a.OPF_sol()
    sol.running_time = time.time() - t1
    sol.gurobi_model = m

    if u.gurobi_handle_errors(m, algname=alg):
        sol.obj = obj.getValue()
        sol.x = {k: x[k].x for k in range(ins.n)}
        sol.idx = [k for k in ins.I if x[k].x == 1]
        sol.l = l
        sol.P = P
        sol.Q = Q
        sol.v = v
        logging.info("\tx            = %s" % str(sol.x))
        logging.info("\t{k: x_k>0}   = %s" % str([k for k in range(ins.n) if sol.x[k] > 0]))
        sol.succeed = True
    else:
        sol.succeed = False

    return sol

# used for TCNS 2016 paper
def max_OPF_OPT(ins, cons='', tolerance=0.0001, debug=False):
    t1 = time.time()

    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)

    x = [0] * ins.n
    #    dummy_p = {i: 0 for i in T.nodes()}
    #    dummy_q = {i: 0 for i in T.nodes()}
    v = {i: 0 for i in T.nodes()}
    v[0] = ins.v_0
    l = {e: 0 for e in T.edges()}
    P = {e: 0 for e in T.edges()}
    Q = {e: 0 for e in T.edges()}
    for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % k)
    for k in ins.F: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    #    for k in T.nodes(): dummy_p[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_p[%d]" % k)
    #    for k in T.nodes(): dummy_q[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_q[%d]" % k)
    for i in T.nodes()[1:]:
        v[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % i)
    for e in T.edges(): l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
    for e in T.edges(): P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
    for e in T.edges(): Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    m.update()

    obj = gbp.quicksum(x[k] * ins.loads_utilities[k] for k in range(ins.n))
    m.setObjective(obj, gbp.GRB.MAXIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']
        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i
        # m.addConstr(dummy_p[e[1]] >= 0, "dummy_P_%d" % e[1])
        # m.addConstr(dummy_q[e[1]] >= 0, "dummy_Q_%d"%e[1])
        rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [P[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])  # + dummy_p[e[1]]
        rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])  # + dummy_q[e[1]]
        # m.addQConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s=" % str(e))
        m.addQConstr(rhs_P - tolerance <= P[e], "P_%s=" % str(e))
        m.addQConstr(rhs_P + tolerance >= P[e], "P_%s=" % str(e))
        # m.addQConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))
        m.addQConstr(rhs_Q - tolerance <= Q[e] <= rhs_Q + tolerance, "Q_%s=" % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d=" % e[1])

        if cons == 'C' or cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if cons == 'V' or cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d" % e[1])  # voltage constraint
    for i in ins.F:
        m.addConstr(x[i] <= 1, "x[%d]: ub")
        m.addConstr(x[i] >= 0, "x[%d]: lb")
    m.update()
    m.optimize()

    u.gurobi_handle_errors(m)

    if logging.getLogger().getEffectiveLevel() == logging.INFO:
        for e in T.edges():
            z = T[e[0]][e[1]]['z']
            logging.info('\t==== Edge: %s ===' % str(e))
            S_ = np.sqrt(P[e].x ** 2 + Q[e].x ** 2)
            logging.info('\t|S|          = %09.6f\t\t(diff = %e)' % (S_, S_ - T[e[0]][e[1]]['C']))
            logging.info('\tl            = %09.6f\t\t' \
                         '(diff = %e)' % (l[e].x, l[e].x - (P[e].x ** 2 + Q[e].x ** 2) / v[0]))
            loss = l[e].x * np.sqrt(T[e[0]][e[1]]['z'][0] ** 2 + T[e[0]][e[1]]['z'][1] ** 2)
            logging.info('\tLoss         = %09.6f' % (loss))
            pure_demand_P = np.sum([ins.loads_P[k] * x[k].x for k in T[e[0]][e[1]]['K']])
            pure_demand_Q = np.sum([ins.loads_Q[k] * x[k].x for k in T[e[0]][e[1]]['K']])
            total_loss_P = P[e].x - pure_demand_P  # - dummy_p[e[1]].x
            total_loss_Q = Q[e].x - pure_demand_Q  # - dummy_q[e[1]].x
            T[e[0]][e[1]]['L'] = (total_loss_P, total_loss_Q)
            logging.info('\tTotal loss   = %09.6f' % np.sqrt(total_loss_P ** 2 + total_loss_Q ** 2))
            logging.info('\tPure demand  = %09.6f' % np.sqrt(pure_demand_P ** 2 + pure_demand_Q ** 2))

            # dummy = np.sqrt(dummy_p[e[1]].x ** 2 + dummy_q[e[1]].x ** 2)
            # logging.info('\t|dummy_S|    = %09.6f\t\tP,Q = (%09.6f,%09.6f)' % (dummy, dummy_p[e[1]].x, dummy_q[e[1]].x))

            if (S_ != 0): logging.info('\tloss to pow. = %09.6f' % (loss / S_))
            logging.info('\tloss to cap. = %09.6f' % (loss / T[e[0]][e[1]]['C']))

            logging.info('\t=== voltages ===')
        for i in T.nodes()[1:]:
            logging.info('\tv_%3d        = %f\t\t(diff = %e)' % (i, v[i].x, v[i].x - ins.v_min))
    if debug:
        for e in T.edges():
            print '\t===== Edge: %s =====' % str(e)
            S_ = np.sqrt(P[e].x ** 2 + Q[e].x ** 2)
            print '\tC            = %09.6f' % (T[e[0]][e[1]]['C'])
            print '\t|S|          = %09.6f\t\t      (diff = %.10f)' % (S_, S_ - T[e[0]][e[1]]['C'])
            pure_demand_P = np.sum([ins.loads_P[k] * x[k].x for k in T[e[0]][e[1]]['K']])
            pure_demand_Q = np.sum([ins.loads_Q[k] * x[k].x for k in T[e[0]][e[1]]['K']])
            # pure_demand_P  = np.sum([x[k].x * ins.loads_P[k] for k in T.node[e[1]]['N']]) + np.sum(
            # [P[(e[1], h)].x for h in T.edge[e[1]].keys() if e[1] < h]) #+ dummy_p[e[1]]
            # pure_demand_Q  = np.sum([x[k].x * ins.loads_Q[k] for k in T.node[e[1]]['N']]) + np.sum(
            #    [Q[(e[1], h)].x for h in T.edge[e[1]].keys() if e[1] < h]) #+ dummy_p[e[1]]
            pure_S = np.sqrt(pure_demand_P ** 2 + pure_demand_Q ** 2)
            print '\tpure |S|    = %09.6f\t\t' % (pure_S)

            print '\tP,Q          = %09.6f, %09.6f' % (P[e].x, Q[e].x)

    sol = a.OPF_sol()
    sol.obj = obj.getValue()
    sol.x = {k: x[k].x for k in range(ins.n)}
    sol.idx = [k for k in ins.I if x[k].x == 1]
    sol.status = m
    # sol.l = l.items()
    # sol.P = P.items()
    # sol.Q = Q.items()
    # sol.v = v.items()
    logging.info("\tx            = %s" % str(sol.x))
    logging.info("\t{k: x_k>0}   = %s" % str([k for k in range(ins.n) if sol.x[k] > 0]))

    sol.running_time = time.time() - t1

    return sol


def max_OPF_OPT_fixed(ins, X, cons='', tolerance=0.0001):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)

    x = [0] * ins.n
    v = {i: 0 for i in T.nodes()}
    v[0] = ins.v_0
    l = {e: 0 for e in T.edges()}
    P = {e: 0 for e in T.edges()}
    Q = {e: 0 for e in T.edges()}
    for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % k)
    for k in ins.F: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    for i in T.nodes()[1:]:
        v[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % i)
    for e in T.edges(): l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
    for e in T.edges(): P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
    for e in T.edges(): Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    m.update()

    obj = gbp.quicksum(x[k] * ins.loads_utilities[k] for k in range(ins.n))
    m.setObjective(obj, gbp.GRB.MAXIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']

        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i
        rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [P[(e[1], h)] for h in T.edge[e[1]].keys()])  # + dummy_p[e[1]]
        rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys()])  # + dummy_q[e[1]]
        m.addQConstr(rhs_P - tolerance <= P[e], "P_%s=" % str(e))
        m.addQConstr(rhs_P + tolerance >= P[e], "P_%s=" % str(e))
        m.addQConstr(rhs_Q - tolerance <= Q[e] <= rhs_Q + tolerance, "Q_%s=" % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d=" % e[1])

        if cons == 'C' or cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if cons == 'V' or cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d" % e[1])  # voltage constraint
    for i in ins.F:
        m.addConstr(x[i], gbp.GRB.EQUAL, X[i], "x[%d]: ub")
    m.update()
    m.optimize()

    sol = a.OPF_sol()
    u.gurobi_handle_errors(m)
    if (m.status in [3, 12]):  # infeasible or numeric error
        sol.obj = -np.inf
        return sol
    else:
        sol.obj = obj.getValue()
    sol.status = m

    sol.running_time = time.time() - t1

    return sol


def min_loss_OPF(ins, x, cons='', tolerance=0.0001, v_tolerance=0.1):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)
    # m.setParam("NumericFocus",3) #0 automatic, 1-3 means how hard gurobi check numeric accuracty
    # m.setParam("IntFeasTol", 0.01)
    # m.setParam("MIPGapAbs", 0.01)


    # dummy_p = {i: 0 for i in T.nodes()}
    # dummy_q = {i: 0 for i in T.nodes()}
    v = {i: 0 for i in T.nodes()}
    v[0] = ins.v_0
    l = {e: 0 for e in T.edges()}
    P = {e: 0 for e in T.edges()}
    Q = {e: 0 for e in T.edges()}
    # for k in T.nodes(): dummy_p[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_p[%d]" % k)
    # for k in T.nodes(): dummy_q[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_q[%d]" % k)
    for k in T.nodes()[1:]:
        v[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % k)
    for e in T.edges(): l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
    for e in T.edges(): P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
    for e in T.edges(): Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    m.update()

    obj = gbp.quicksum([l[e] * np.sqrt(T[e[0]][e[1]]['z'][0] ** 2 + T[e[0]][e[1]]['z'][0] ** 2) for e in T.edges()])
    # obj = gbp.quicksum([l[e] for e in T.edges()])

    m.setObjective(obj, gbp.GRB.MINIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']

        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i
        # m.addConstr(dummy_p[e[1]] >= -tolerance, "dummy_P_%d" % e[1])
        # m.addConstr(tolerance >= dummy_q[e[1]] >= -tolerance, "dummy_P_%d" % e[1])

        # index_set = set(T.node[e[1]]['N']).intersection(set(idx))
        rhs_P = l[e] * z[0] + gbp.quicksum([ins.loads_P[k] * x[k] for k in T.node[e[1]]['N']]) + gbp.quicksum(
            [P[(e[1], h)] for h in T.edge[e[1]].keys()])  # + dummy_p[e[1]]
        rhs_Q = l[e] * z[1] + gbp.quicksum([ins.loads_Q[k] * x[k] for k in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys()])  # + dummy_q[e[1]]
        # m.addQConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s=" % str(e))
        m.addQConstr(rhs_P + tolerance >= P[e], "P_%s=" % str(e))
        m.addQConstr(P[e] >= rhs_P - tolerance, "P_%s=" % str(e))
        # m.addQConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))
        m.addQConstr(rhs_Q + tolerance >= Q[e], "Q_%s=" % str(e))
        m.addQConstr(Q[e] >= rhs_Q - tolerance, "Q_%s=" % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        # m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d=" % e[1])
        m.addConstr(v[e[1]] >= rhs_v - v_tolerance, "l_v_%d=" % e[1])
        m.addConstr(rhs_v + v_tolerance >= v[e[1]], "r_v_%d=" % e[1])

        if cons == 'C' or cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if cons == 'V' or cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d" % e[1])  # voltage constraint

    m.update()
    m.optimize()
    sol = a.OPF_sol()
    sol.status = m.status
    sol.m = m
    sol.running_time = time.time() - t1

    # u.gurobi_handle_errors(m)

    if (m.status in [3, 12]):  # infeasible or numeric error
        # logging.warning("Infeasible!")
        sol.obj = -np.inf
        return sol
    else:
        sol.obj = obj.getValue()

    if logging.getLogger().getEffectiveLevel() == logging.INFO:
        for e in T.edges():
            z = T[e[0]][e[1]]['z']
            logging.info('\t==== Edge: %s ===' % str(e))
            S_ = np.sqrt(P[e].x ** 2 + Q[e].x ** 2)
            logging.info('\t|S|          = %09.6f\t\t(diff = %e)' % (S_, S_ - T[e[0]][e[1]]['C']))
            logging.info(
                '\tl            = %09.6f\t\t(diff = %e)' % (l[e].x, l[e].x - (P[e].x ** 2 + Q[e].x ** 2) / v[0]))
            loss = l[e].x * np.sqrt(T[e[0]][e[1]]['z'][0] ** 2 + T[e[0]][e[1]]['z'][1] ** 2)
            logging.info('\tLoss         = %09.6f' % (loss))

            pure_demands_P = [ins.loads_P[k] * x[k] for k in T[e[0]][e[1]]['K']]
            pure_demands_Q = [ins.loads_Q[k] * x[k] for k in T[e[0]][e[1]]['K']]
            total_loss_P = P[e].x - sum(pure_demands_P) - dummy_p[e[1]].x
            total_loss_Q = Q[e].x - sum(pure_demands_Q) - dummy_q[e[1]].x
            T[e[0]][e[1]]['L'] = (total_loss_P, total_loss_Q)
            logging.info('\tTotal loss   = %09.6f' % np.sqrt(total_loss_P ** 2 + total_loss_Q ** 2))
            logging.info('\tPure demand  = %09.6f' % np.sqrt(sum(pure_demands_P) ** 2 + sum(pure_demands_Q) ** 2))

            dummy = np.sqrt(dummy_p[e[1]].x ** 2 + dummy_q[e[1]].x ** 2)
            logging.info('\t|dummy_S|    = %09.6f\t\tP,Q = (%09.6f,%09.6f)' % (dummy, dummy_p[e[1]].x, dummy_q[e[1]].x))

            logging.info('\tloss to pow. = %09.6f' % (loss / S_))
            logging.info('\tloss to cap. = %09.6f' % (loss / T[e[0]][e[1]]['C']))
        logging.info('\t=== voltages ===')
        for k in T.nodes()[1:]:
            logging.info('\tv_%3d        = %f\t\t(diff = %e)' % (k, v[k].x, v[k].x - ins.v_min))

    return sol


if __name__ == "__main__":
    # ins = a.rnd_tree_instance(n=100, depth=5, branch=2, capacity_range=(2, 4), util_func=lambda x,y: x-x + x**2)
    # ins = a.rnd_path_instance(n=10,node_count=3, capacity_range=(1, 1))
    # ins.F = ins.I
    # ins.I = []
    ins = a.single_link_instance(n=10, capacity=4.6, z=(.001, .001))
    # angle = 35 * np.pi/180
    # ins.loads_P = np.random.uniform(0,1,ins.n)
    # ins.loads_Q = np.array([r*np.sin(angle)*(-1)**np.random.randint(0,2) for r in ins.loads_P])
    # ins.loads_Q = np.zeros(ins.n)
    # ins.loads_utilities = [np.sqrt(ins.loads_P[k]**2 + ins.loads_Q[k]**2) for k in range(ins.n)]
    #
    # ins.loads_S = ins.loads_P
    # ins.loads_utilities = ins.loads_P
    # ins.F = ins.I
    # ins.I = []

    a.print_instance(ins)

    #    print '############## maxOPF ##############'
    #    sol2 = max_OPF(ins, cons='C')
    #    print 'obj value: ', sol2.obj
    #    print 'obj idx: ', sol2.idx
    #    print 'time: ', sol2.running_time

    print '############## minOPF ##############'
    sol2 = min_loss_OPF(ins, idx=[0, 1, 2, 3, 4, 5, 6, 7, 8], cons='C')
    print 'obj value: ', sol2.obj
    print 'time: ', sol2.running_time

# T = a.network_38node()
#    max_loss(T, (0,2), cons='C')
