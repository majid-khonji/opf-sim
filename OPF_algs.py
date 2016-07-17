#!/usr/bin/python
__author__ = 'mkhonji'
import numpy as np
import time
import instance as a
import logging
import networkx as nx
import util as u

try:
    import gurobipy as gbp
except ImportError:
    logging.warning("Grubi not available!!")


def max_OPF_OPT(ins, cons=''):
    t1 = time.time()

    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)

    x = [0] * ins.n
    dummy_p = {i: 0 for i in T.nodes()}
    dummy_q = {i: 0 for i in T.nodes()}
    v = {i: 0 for i in T.nodes()}
    v[0] = ins.v_0
    l = {e: 0 for e in T.edges()}
    P = {e: 0 for e in T.edges()}
    Q = {e: 0 for e in T.edges()}
    for k in ins.I: x[k] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % k)
    for k in ins.F: x[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[%d]" % k)
    for k in T.nodes(): dummy_p[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_p[%d]" % k)
    for k in T.nodes(): dummy_q[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_q[%d]" % k)
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
        m.addConstr(dummy_p[e[1]] >= 0, "dummy_P_%d" % e[1])
        # m.addConstr(dummy_q[e[1]] >= 0, "dummy_Q_%d"%e[1])
        rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [P[(e[1], h)] for h in T.edge[e[1]].keys() ]) + dummy_p[e[1]]
        rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys() ]) + dummy_q[e[1]]
        m.addQConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s=" % str(e))
        m.addQConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))

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
            total_loss_P = P[e].x - pure_demand_P - dummy_p[e[1]].x
            total_loss_Q = Q[e].x - pure_demand_Q - dummy_q[e[1]].x
            T[e[0]][e[1]]['L'] = (total_loss_P, total_loss_Q)
            logging.info('\tTotal loss   = %09.6f' % np.sqrt(total_loss_P ** 2 + total_loss_Q ** 2))
            logging.info('\tPure demand  = %09.6f' % np.sqrt(pure_demand_P ** 2 + pure_demand_Q ** 2))

            dummy = np.sqrt(dummy_p[e[1]].x ** 2 + dummy_q[e[1]].x ** 2)
            logging.info('\t|dummy_S|    = %09.6f\t\tP,Q = (%09.6f,%09.6f)' % (dummy, dummy_p[e[1]].x, dummy_q[e[1]].x))

            if (S_ != 0): logging.info('\tloss to pow. = %09.6f' % (loss / S_))
            logging.info('\tloss to cap. = %09.6f' % (loss / T[e[0]][e[1]]['C']))

            logging.info('\t=== voltages ===')
        for i in T.nodes()[1:]:
            logging.info('\tv_%3d        = %f\t\t(diff = %e)' % (i, v[i].x, v[i].x - ins.v_min))

    sol = a.maxOPF_sol()
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


def min_loss_OPF(ins, x, cons=''):
    t1 = time.time()
    T = ins.topology
    m = gbp.Model("qcp")
    u.gurobi_setting(m)

    dummy_p = {i: 0 for i in T.nodes()}
    dummy_q = {i: 0 for i in T.nodes()}
    v = {i: 0 for i in T.nodes()}
    v[0] = ins.v_0
    l = {e: 0 for e in T.edges()}
    P = {e: 0 for e in T.edges()}
    Q = {e: 0 for e in T.edges()}
    for k in T.nodes(): dummy_p[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_p[%d]" % k)
    for k in T.nodes(): dummy_q[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="dummy_q[%d]" % k)
    for k in T.nodes()[1:]:
        v[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % k)
    for e in T.edges(): l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
    for e in T.edges(): P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
    for e in T.edges(): Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    m.update()

    obj = gbp.quicksum([l[e] * np.sqrt(T[e[0]][e[1]]['z'][0] ** 2 + T[e[0]][e[1]]['z'][0] ** 2) for e in T.edges()])
    m.setObjective(obj, gbp.GRB.MINIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']

        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i
        m.addConstr(dummy_p[e[1]] >= 0, "dummy_P_%d" % e[1])

        # index_set = set(T.node[e[1]]['N']).intersection(set(idx))
        rhs_P = l[e] * z[0] + gbp.quicksum([ins.loads_P[k] * x[k] for k in T.node[e[1]]['N']]) + gbp.quicksum(
            [P[(e[1], h)] for h in T.edge[e[1]].keys() ]) + dummy_p[e[1]]
        rhs_Q = l[e] * z[1] + gbp.quicksum([ins.loads_Q[k] * x[k] for k in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys() ]) + dummy_q[e[1]]
        m.addQConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s=" % str(e))
        m.addQConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d=" % e[1])

        if cons == 'C' or cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if cons == 'V' or cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d" % e[1])  # voltage constraint

    m.update()
    m.optimize()
    sol = a.maxOPF_sol()
    sol.status = m.status
    sol.m = m
    sol.running_time = time.time() - t1

    if (m.status == 3):
        # logging.warning("Infeasible!")
        sol.obj = -np.inf
        return sol
    # if (u.gurobi_handle_errors(m) == False):
    #     sol.obj = -np.inf
    #     return sol
    #sol.obj = obj.getValue()

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


# Erroneous
def max_loss(topology, e=(0, 1), v_0=1, v_min=.82, cons=''):
    t1 = time.time()

    m = gbp.Model("qcp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 200)
    m.setParam("MIPGapAbs", 0)
    m.setParam("IntFeasTol", 1e-9)  # for integrality difference
    m.setParam("FeasibilityTol", 1e-9)

    T = topology

    # obtain children nodes of e[1] (including e[1])
    children_nodes = None
    T_ = topology.copy()
    if e[0] != 0:  # not root, then delete the dege connected to e[0]
        # disconnect the graph at e[0]
        T_.remove_node(e[0])
        T_ = nx.bfs_tree(T_, e[1])
        children_nodes = T_.nodes()
    else:
        children_nodes = T_.nodes()[1:]

    p = {k: 0 for k in children_nodes}
    q = {k: 0 for k in children_nodes}

    v = {i: 0 for i in T.nodes()}
    v[0] = v_0
    l = {e: 0 for e in T.edges()}
    P = {e: 0 for e in T.edges()}
    Q = {e: 0 for e in T.edges()}
    for i in children_nodes:
        p[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="p[%d]" % i)
        q[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="q[%d]" % i)
    for i in T.nodes()[1:]:
        v[i] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="v_%d" % i)
    for e in T.edges():
        l[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="l_%s" % str(e))
        P[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(e))
        Q[e] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="Q_%s" % str(e))
    m.update()

    # obj = P[e] + P[e] - T[e[0]][e[1]]['z'][0]*l[e] - T[e[0]][e[1]]['z'][1]*l[e]
    obj = P[e] + Q[e]
    m.setObjective(obj, gbp.GRB.MAXIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']

        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, P[e] * P[e], "l_%s" % str(e))  # l= |P|^2/ v_i

        rhs_P = l[e] * z[0] + gbp.quicksum([P[(e[1], h)] for h in T.edge[e[1]].keys()])
        rhs_Q = l[e] * z[1] + gbp.quicksum([Q[(e[1], h)] for h in T.edge[e[1]].keys()])
        if e[1] in children_nodes:
            m.addConstr(p[e[1]] >= 0, "q_%d" % e[1])
            # m.addConstr(q[e[1]] >= 0, "p_%d"%e[1])
            rhs_P += p[e[1]]
            rhs_Q += q[e[1]]

        m.addQConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s=" % str(e))
        m.addQConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d=" % e[1])

        if cons == 'C' or cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if cons == 'V' or cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, v_min, "v_%d" % e[1])  # voltage constraint

        m.update()
    m.optimize()

    u.gurobi_handle_errors(m)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']
        print '\t===== Edge: %s =====' % str(e)
        S_ = np.sqrt(P[e].x ** 2 + Q[e].x ** 2)
        print '\tC            = %09.6f' % (T[e[0]][e[1]]['C'])
        print '\t|S|          = %09.6f\t\t      (diff = %.10f)' % (S_, S_ - T[e[0]][e[1]]['C'])
        print '\tP,Q          = %09.6f, %09.6f' % (P[e].x, Q[e].x)
        print '\tl            = %09.6f\t\t      (diff = %.10f)' % (l[e].x, l[e].x - (P[e].x ** 2 + Q[e].x ** 2) / v[0])
        loss = l[e].x * np.sqrt(z[0] ** 2 + z[1] ** 2)
        print '\tLoss on e    = %09.6f\t\tR,I = (%09.6f, %09.6f)' % (loss, z[0] * l[e].x, z[1] * l[e].x)
        if e[1] in children_nodes:
            print '\tload on %-2d   = %09.6f\t\tR,I = (%09.6f, %09.6f)' % (
                e[1], np.sqrt(p[e[1]].x ** 2 + q[e[1]].x ** 2), p[e[1]].x, q[e[1]].x)
        pure_demands_P = sum([p[i].x for i in children_nodes])
        pure_demands_Q = sum([q[i].x for i in children_nodes])
        pure_demands_S = np.sqrt(pure_demands_P ** 2 + pure_demands_Q ** 2)
        total_loss_P = P[e].x - pure_demands_P
        total_loss_Q = Q[e].x - pure_demands_Q
        total_loss_S = np.sqrt(total_loss_P ** 2 + total_loss_Q ** 2)
        T[e[0]][e[1]]['L'] = (total_loss_P, total_loss_Q)
        print '\t|Total loss| = %09.6f\t\tR,I = (%09.6f, %09.6f)' % (total_loss_S, total_loss_P, total_loss_Q)
        print '\t|Pure loads| = %09.6f\t\tP,Q = (%09.6f, %09.6f)' % (pure_demands_S, pure_demands_P, pure_demands_Q)

        print '\tloss to pow. = %09.6f' % (total_loss_S / S_)
        print '\tloss to cap. = %09.6f' % (total_loss_S / T[e[0]][e[1]]['C'])

    print '\t=======---------========'
    print '\t======= voltage ========'
    for i in T.nodes()[1:]:
        print '\tv_%3d        = %f\t\t           (diff = %.10f)' % (i, v[i].x, v[i].x - v_min)

    print 'obj value = %.6f' % obj.getValue()
    running_time = time.time() - t1
    print 'finished in %.3f' % running_time, ' secs'



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
