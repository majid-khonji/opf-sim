#!/usr/bin/python
__author__ = 'Majid Khonji'
import numpy as np
import time
import instance as a
import logging
import networkx as nx
import util as u
import itertools
import copy

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
            guess_I2 = {k: 0 for k in ins.I if ins.loads_utilities[k] <= min_util}
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
                sol.frac_comp_percentage = sol.frac_comp_count / (ins.n * 1.) * 100
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
                        obj += T.graph['S_base'] * u.obj_min_loss_penalty(ins, sol, output='loss')
                        sol2.x = sol.x
                        sol2.succeed = True
                    sol2.obj = obj
                    sol2.frac_comp_count = sol.frac_comp_count
                    sol2.frac_comp_percentage = sol2.frac_comp_count / (ins.n * 1.) * 100

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
            m.addConstr(lhs_P, gbp.GRB.LESS_EQUAL, C_P, "Cp_%s" % str(e))

            C_Q = np.sum([sol.x[k] * ins.loads_Q[k] for k in edge_customers])
            lhs_Q = gbp.quicksum([x[k] * ins.loads_Q[k] for k in edge_customers])
            m.addConstr(lhs_Q, gbp.GRB.LESS_EQUAL, C_Q, "Cq_%s" % str(e))
    if not ins.drop_l_terms and (ins.cons == '' or ins.cons == 'V'):
        for l in ins.leaf_nodes:
            C_V = np.sum([ins.Q[(k, l)] * sol.x[k] for k in customers])
            lhs_V = gbp.quicksum([ins.Q[(k, l)] * x[k] for k in customers])
            m.addConstr(lhs_V, gbp.GRB.LESS_EQUAL, C_V, "Cv_%s" % str(l))
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
    obj = gbp.quicksum([T[e[0]][e[1]]['z'][0] * l[e] for e in T.edges()])
    m.setObjective(obj, gbp.GRB.MINIMIZE)
    for e in T.edges():
        z = T[e[0]][e[1]]['z']
        subtree_edges = nx.bfs_edges(T, e[1])
        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i
        # rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [P[(e[1], h)] for h in T.edge[e[1]].keys() ])
        # rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [P[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])
        rhs_P = l[e] * z[0] + gbp.quicksum([x[k] * ins.loads_P[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h] * z[0] for h in subtree_edges])
        m.addConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s " % str(e))

        # rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [Q[(e[1], h)] for h in T.edge[e[1]].keys()])
        # rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [Q[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])
        rhs_Q = l[e] * z[1] + gbp.quicksum([x[k] * ins.loads_Q[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h] * z[1] for h in subtree_edges])
        m.addConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s " % str(e))

        rhs_v = v[e[0]] + (z[0] ** 2 + z[1] ** 2) * l[e] - 2 * (z[0] * P[e] + z[1] * Q[e])
        m.addConstr(v[e[1]], gbp.GRB.EQUAL, rhs_v, "v_%d " % e[1])
        # m.update()

        if ins.cons == 'C' or ins.cons == '':
            m.addQConstr(P[e] * P[e] + Q[e] * Q[e], gbp.GRB.LESS_EQUAL, T[e[0]][e[1]]['C'] ** 2,
                         "C_%s" % str(e))  # capacity constraint
        if ins.cons == 'V' or ins.cons == '':
            m.addConstr(v[e[1]], gbp.GRB.GREATER_EQUAL, ins.v_min, "v_%d bound" % e[1])  # voltage constraint

        m.addConstr(v[e[1]] >= 0, "v_%d+" % e[1])
        m.addConstr(l[e] >= 0, "l_%s+" % str(e))
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

def round_EV_scheduling4(ins, guess_x={}, round_x_after_y=True):

    t1 = time.time()
    sol = max_ev_scheduling_OPT(ins, guess_x=guess_x, fractional=True)
    frac_sol = copy.deepcopy(sol)
    sol.frac_sol = frac_sol

    sol.frac_x_comp_count = 0
    sol.frac_y_comp_count = 0
    sol.rounded_up_count = 0
    sol.rounded_down_count = 0
    sol.count_y_due_to_rounded_x = 0
    sol.count_rounded_up_y_due_to_rounded_x = 0
    sol.customer_satisfy_ratio = {}

    energy_usage = {}
    energy_usage_frac_sol = {}
    if sol.succeed:
        # customers = np.setdiff1d(np.arange(ins.n), guess_x)
        customers = np.arange(ins.n)

        sol.total_ev_charge_at_time_frac_sol = {}
        for t in np.arange(ins.scheduling_horizon):
            sol.total_ev_charge_at_time_frac_sol[t] = np.sum([ins.charging_rates[c] * sol.x[(k, c, t)]
                                                              for k in ins.customers_at_time[t] for c in
                                                              ins.customer_charging_options[k]])
        # calculated energy user of each customer
        for k in customers:
            energy_usage[k] = 0
            energy_usage_frac_sol[k] = 0
            if ins.rounding_tolerance < sol.y[k] < 1 - ins.rounding_tolerance:
                # sol.y[k] = 0
                sol.frac_y_comp_count += 1
                # is_y_rounded = True
            for t in ins.customer_charging_time_path[k]:
                for c in ins.customer_charging_options[k]:
                    energy_usage_frac_sol[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length
                    # if is_y_rounded:
                    #     sol.x[(k, c, t)] = 0

        #### Greedy 4 rounding

        charging_cost = {k:sol.x[(k, c, t)]*ins.cost_rate_matrix[c,t] for k in customers for t in ins.customer_charging_time_path[k] for c in ins.customer_charging_options[k]}
        list_cost= {k:ins.customer_utilities[k]*sol.y[k]- charging_cost[k] for k in customers}
        sorted_customers = [k for k, v in sorted(list_cost.iteritems(), key=lambda (k, v): (v, k), reverse=True)]

        total_ev_power_at_time = {t:0 for t in np.arange(ins.scheduling_horizon)}
        energy_usage = {k:0 for k in customers}

        for k in sorted_customers:
            time_list_cost= {(c,t): sol.x[k,c,t] for c in ins.customer_charging_options[k] for t in ins.customer_charging_time_path[k]}
            sorted_time_option = {r:v for r, v in sorted(time_list_cost.iteritems(), key=lambda (r, v): (v, r), reverse=True) if v != 0}
            for (c,t) in sorted_time_option:
                if ins.rounding_tolerance < sol.x[(k, c, t)] < 1 - ins.rounding_tolerance:
                    sol.frac_x_comp_count += 1

                other_chg_options = np.setdiff1d(ins.customer_charging_options[k], [c])
                total_ev_power_after_rounding_up = total_ev_power_at_time[t] + ins.charging_rates[c]

                usage_after_rounding_up = energy_usage[k] + ins.charging_rates[c]*ins.step_length
                usage_after_rounding_down = energy_usage[k]
                if total_ev_power_after_rounding_up <= (
                            ins.capacity_over_time[t] - ins.base_load_over_time[t]) and usage_after_rounding_down < \
                        ins.customer_usage[k] :
                    sol.x[k,c,t] = 1
                    energy_usage[k] = usage_after_rounding_up
                    total_ev_power_at_time[t] = total_ev_power_after_rounding_up

                    # print('   + x[%d,%d,%d] rounded up'%(k,c,t), sol.x[k,c,t])
                    # print('  --- x[%d,%d,%d] rounded down'%(k,c_,t), sol.x[k,c_,t])
                else:
                    sol.x[k,c, t] = 0
                    # print('   + x[%d,%d,%d] rounded down'%(k,c,t), sol.x[k,c,t])

                if sol.x[k,c,t] >= 1-ins.rounding_tolerance:
                    for c_ in other_chg_options:
                        sol.x[(k, c_, t)] = 0



            # set y variable
            sol.customer_satisfy_ratio[k] = energy_usage[k]/ins.customer_usage[k]
            if energy_usage[k] < ins.customer_usage[k] - ins.rounding_tolerance:
                sol.y[k] = 0
                sol.count_y_due_to_rounded_x += 1
                for c in ins.customer_charging_options[k]:
                    for t in ins.customer_charging_time_path[k]:
                        energy_usage[k] -= sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length
                        sol.x[(k, c, t)] = 0
            else:
                sol.y[k]=1
                sol.count_rounded_up_y_due_to_rounded_x += 1

        sol.running_time = time.time() - t1

        # calculate new obj
        obj = 0
        for k in np.arange(ins.n):
            obj += ins.customer_utilities[k] * sol.y[k]
            for c in ins.customer_charging_options[k]:
                for t in ins.customer_charging_time_path[k]:
                    obj -= sol.x[(k, c, t)] * ins.cost_rate_matrix[c, t]
        # calculate P
        P = {}
        for t in np.arange(ins.scheduling_horizon):
            P[t] = ins.base_load_over_time[t]
            for k in ins.customers_at_time[t]:
                for c in ins.customer_charging_options[k]:
                    P[t] += sol.x[(k, c, t)] * ins.charging_rates[c]

        sol.frac_obj = sol.obj
        sol.obj = obj
        sol.customer_energy_usage = energy_usage
        sol.customer_energy_usage_frac_sol = energy_usage_frac_sol
        sol.P = P
        sol.ar = sol.obj/sol.frac_obj
    sol.running_time = time.time() - t1
    return sol

def round_EV_scheduling3(ins, guess_x={}, round_x_after_y=True):

    t1 = time.time()
    sol = max_ev_scheduling_OPT(ins, guess_x=guess_x, fractional=True)
    frac_sol = copy.deepcopy(sol)
    sol.frac_sol = frac_sol

    sol.frac_x_comp_count = 0
    sol.frac_y_comp_count = 0
    sol.rounded_up_count = 0
    sol.rounded_down_count = 0
    sol.count_y_due_to_rounded_x = 0
    sol.count_rounded_up_y_due_to_rounded_x = 0
    sol.customer_satisfy_ratio = {}

    energy_usage = {}
    energy_usage_frac_sol = {}
    if sol.succeed:
        # customers = np.setdiff1d(np.arange(ins.n), guess_x)
        customers = np.arange(ins.n)

        sol.total_ev_charge_at_time_frac_sol = {}
        for t in np.arange(ins.scheduling_horizon):
            sol.total_ev_charge_at_time_frac_sol[t] = np.sum([ins.charging_rates[c] * sol.x[(k, c, t)]
                                                              for k in ins.customers_at_time[t] for c in
                                                              ins.customer_charging_options[k]])
        # calculated energy user of each customer
        for k in customers:
            energy_usage[k] = 0
            energy_usage_frac_sol[k] = 0
            if ins.rounding_tolerance < sol.y[k] < 1 - ins.rounding_tolerance:
                # sol.y[k] = 0
                sol.frac_y_comp_count += 1
                # is_y_rounded = True
            for t in ins.customer_charging_time_path[k]:
                for c in ins.customer_charging_options[k]:
                    energy_usage_frac_sol[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length
                    # if is_y_rounded:
                    #     sol.x[(k, c, t)] = 0

        #### Greedy 3 rounding

        charging_cost = {k:sol.x[(k, c, t)]*ins.cost_rate_matrix[c,t] for k in customers for t in ins.customer_charging_time_path[k] for c in ins.customer_charging_options[k]}
        list_cost= {k:ins.customer_utilities[k]*sol.y[k]- charging_cost[k] for k in customers}
        sorted_customers = [k for k, v in sorted(list_cost.iteritems(), key=lambda (k, v): (v, k), reverse=True)]

        total_ev_power_at_time = {t:0 for t in np.arange(ins.scheduling_horizon)}
        energy_usage = {k:0 for k in customers}

        for k in sorted_customers:
            for t in ins.customer_charging_time_path[k]:
                options = [sol.x[k, cc, t] for cc in ins.customer_charging_options[k]]
                max_idx = np.argmax(options)
                c = ins.customer_charging_options[k][max_idx]
                if ins.rounding_tolerance < sol.x[(k, c, t)] < 1 - ins.rounding_tolerance:
                    sol.frac_x_comp_count += 1

                other_chg_options = np.setdiff1d(ins.customer_charging_options[k], [c])
                total_ev_power_after_rounding_up = total_ev_power_at_time[t] + ins.charging_rates[c]

                usage_after_rounding_up = energy_usage[k] + ins.charging_rates[c]*ins.step_length
                usage_after_rounding_down = energy_usage[k]
                if total_ev_power_after_rounding_up <= (
                            ins.capacity_over_time[t] - ins.base_load_over_time[t]) and usage_after_rounding_down < \
                        ins.customer_usage[k] :
                    sol.x[k,c,t] = 1
                    energy_usage[k] = usage_after_rounding_up
                    total_ev_power_at_time[t] = total_ev_power_after_rounding_up

                    # print('   + x[%d,%d,%d] rounded up'%(k,c,t), sol.x[k,c,t])
                        # print('  --- x[%d,%d,%d] rounded down'%(k,c_,t), sol.x[k,c_,t])
                else:
                    sol.x[k,c, t] = 0
                    # print('   + x[%d,%d,%d] rounded down'%(k,c,t), sol.x[k,c,t])

                if sol.x[k,c,t] >= 1-ins.rounding_tolerance:
                    for c_ in other_chg_options:
                        sol.x[(k, c_, t)] = 0



            # set y variable
            sol.customer_satisfy_ratio[k] = energy_usage[k]/ins.customer_usage[k]
            if energy_usage[k] < ins.customer_usage[k] - ins.rounding_tolerance:
                sol.y[k] = 0
                sol.count_y_due_to_rounded_x += 1
                for c in ins.customer_charging_options[k]:
                    for t in ins.customer_charging_time_path[k]:
                        energy_usage[k] -= sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length
                        sol.x[(k, c, t)] = 0
            else:
                sol.y[k]=1
                sol.count_rounded_up_y_due_to_rounded_x += 1

        sol.running_time = time.time() - t1

        # calculate new obj
        obj = 0
        for k in np.arange(ins.n):
            obj += ins.customer_utilities[k] * sol.y[k]
            for c in ins.customer_charging_options[k]:
                for t in ins.customer_charging_time_path[k]:
                    obj -= sol.x[(k, c, t)] * ins.cost_rate_matrix[c, t]
        # calculate P
        P = {}
        for t in np.arange(ins.scheduling_horizon):
            P[t] = ins.base_load_over_time[t]
            for k in ins.customers_at_time[t]:
                for c in ins.customer_charging_options[k]:
                    P[t] += sol.x[(k, c, t)] * ins.charging_rates[c]

        sol.frac_obj = sol.obj
        sol.obj = obj
        sol.customer_energy_usage = energy_usage
        sol.customer_energy_usage_frac_sol = energy_usage_frac_sol
        sol.P = P
        sol.ar = sol.obj/sol.frac_obj
    sol.running_time = time.time() - t1
    return sol
def round_EV_scheduling2(ins, guess_x={}, round_x_after_y=True):

    t1 = time.time()
    sol = max_ev_scheduling_OPT(ins, guess_x=guess_x, fractional=True)
    frac_sol = copy.deepcopy(sol)
    sol.frac_sol = frac_sol

    sol.frac_x_comp_count = 0
    sol.frac_y_comp_count = 0
    sol.rounded_up_count = 0
    sol.rounded_down_count = 0
    energy_usage = {}
    energy_usage_frac_sol = {}
    if sol.succeed:
        # customers = np.setdiff1d(np.arange(ins.n), guess_x)
        customers = np.arange(ins.n)

        sol.total_ev_charge_at_time_frac_sol = {}
        for t in np.arange(ins.scheduling_horizon):
            sol.total_ev_charge_at_time_frac_sol[t] = np.sum([ins.charging_rates[c] * sol.x[(k, c, t)]
                                                              for k in ins.customers_at_time[t] for c in
                                                              ins.customer_charging_options[k]])
        # round y then all its x's
        # calculated energy user of each customer
        for k in customers:
            energy_usage[k] = 0
            energy_usage_frac_sol[k] = 0
            if ins.rounding_tolerance < sol.y[k] < 1 - ins.rounding_tolerance:
                # sol.y[k] = 0
                sol.frac_y_comp_count += 1
                # is_y_rounded = True
            for t in ins.customer_charging_time_path[k]:
                for c in ins.customer_charging_options[k]:
                    energy_usage_frac_sol[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length
                    # if is_y_rounded:
                    #     sol.x[(k, c, t)] = 0
                    energy_usage[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length

        #### Greedy rounding
        total_ev_power_at_time = sol.total_ev_charge_at_time_frac_sol.copy()
        for t in np.arange(ins.scheduling_horizon):
            # max_chg_option = {}
            # for k in ins.customers_at_time[t]:
            #     options = [sol.x[k, cc, t] for cc in ins.customer_charging_options[k]]
            #     max_idx = np.argmax(options)
            #     max_chg_option[k] = ins.customer_charging_options[k][max_idx]
            # sort customers max x
            list_x= {(k,c): sol.x[(k, c, t)] for k in ins.customers_at_time[t] for c in ins.customer_charging_options[k]}
            sorteted_ev_option_pair = [k for k, v in sorted(list_x.iteritems(), key=lambda (k, v): (v, k), reverse=True) if v not in [0,1]]
            # sorteted_ev_option_pair = [k for k, v in sorted(list_x.iteritems(), key=lambda (k, v): (v, k), reverse=True)]
            # print 'customers at time ',t, ins.customers_at_time[t]
            # print 'customers at time ',t, ins.customers_at_time[t]
            # print 'sorted customers at time ',t,sorted(list_x.iteritems(), key=lambda (k, v): (v, k), reverse=True)
            # print 'sorted', sorteted_ev_option_pair
            for (k,c) in sorteted_ev_option_pair:
                if ins.rounding_tolerance < sol.x[(k, c, t)] < 1 - ins.rounding_tolerance:
                    sol.frac_x_comp_count += 1
                    # print((k,c,t), 'enetered', sol.x[k,c,t])

                    other_chg_options = np.setdiff1d(ins.customer_charging_options[k], [c])
                    rem_option_power = np.sum([sol.x[(k, c_, t)] * ins.charging_rates[c]  for c_ in
                                               other_chg_options])
                    total_power_after_rounding = total_ev_power_at_time[t] + (1 - sol.x[k, c, t]) * \
                                                                              ins.charging_rates[c] - rem_option_power

                    usage_after_rounding_down = energy_usage[k] - sol.x[k,c , t] * ins.charging_rates[c] * ins.step_length
                    if total_power_after_rounding <= (
                                ins.capacity_over_time[t] - ins.base_load_over_time[t]) and usage_after_rounding_down < \
                            ins.customer_usage[k] :
                        sol.x[k,c,t] = 1
                        # print('   + x[%d,%d,%d] rounded up'%(k,c,t), sol.x[k,c,t])
                        for c_ in other_chg_options:
                            sol.x[(k, c_, t)] = 0
                            # print('  --- x[%d,%d,%d] rounded down'%(k,c_,t), sol.x[k,c_,t])
                    else:
                        sol.x[k,c, t] = 0
                        # print('   + x[%d,%d,%d] rounded down'%(k,c,t), sol.x[k,c,t])
                        energy_usage[k] = usage_after_rounding_down
                        total_power_after_rounding = total_ev_power_at_time[t] - sol.x[k, c, t] * ins.charging_rates[c]

                    total_ev_power_at_time[t] = total_power_after_rounding

        sol.count_y_due_to_rounded_x = 0
        sol.count_rounded_up_y_due_to_rounded_x = 0
        sol.customer_satisfy_ratio = {}
        for k in customers:
            energy_usage[k] = 0
            for t in ins.customer_charging_time_path[k]:
                for c in ins.customer_charging_options[k]:
                    energy_usage[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length

        for k in customers:
            # set y variable
            sol.customer_satisfy_ratio[k] = energy_usage[k]/ins.customer_usage[k]
            if energy_usage[k] < ins.customer_usage[k] - ins.rounding_tolerance:
                sol.y[k] = 0
                sol.count_y_due_to_rounded_x += 1
                for c in ins.customer_charging_options[k]:
                    for t in ins.customer_charging_time_path[k]:
                        energy_usage[k] -= sol.x[k, c, t] * ins.charging_rates[c]
                        sol.x[(k, c, t)] = 0
            else:
                sol.y[k]=1
                sol.count_rounded_up_y_due_to_rounded_x += 1

        sol.running_time = time.time() - t1

        # calculate new obj
        obj = 0
        for k in np.arange(ins.n):
            obj += ins.customer_utilities[k] * sol.y[k]
            for c in ins.customer_charging_options[k]:
                for t in ins.customer_charging_time_path[k]:
                    obj -= sol.x[(k, c, t)] * ins.cost_rate_matrix[c, t]
        # calculate P
        P = {}
        for t in np.arange(ins.scheduling_horizon):
            P[t] = ins.base_load_over_time[t]
            for k in ins.customers_at_time[t]:
                for c in ins.customer_charging_options[k]:
                    P[t] += sol.x[(k, c, t)] * ins.charging_rates[c] * ins.step_length

        sol.frac_obj = sol.obj
        sol.obj = obj
        sol.customer_energy_usage = energy_usage
        sol.customer_energy_usage_frac_sol = energy_usage_frac_sol
        sol.P = P
        sol.ar = sol.obj/sol.frac_obj
    sol.running_time = time.time() - t1
    return sol

def round_EV_scheduling1(ins, guess_x={}, round_x_after_y=True):

    t1 = time.time()
    sol = max_ev_scheduling_OPT(ins, guess_x=guess_x, fractional=True)
    frac_sol = copy.deepcopy(sol)
    sol.frac_sol = frac_sol

    sol.frac_x_comp_count = 0
    sol.frac_y_comp_count = 0
    sol.rounded_up_count = 0
    sol.rounded_down_count = 0
    energy_usage = {}
    energy_usage_frac_sol = {}
    if sol.succeed:
        # customers = np.setdiff1d(np.arange(ins.n), guess_x)
        customers = np.arange(ins.n)

        sol.total_ev_charge_at_time_frac_sol = {}
        for t in np.arange(ins.scheduling_horizon):
            sol.total_ev_charge_at_time_frac_sol[t] = np.sum([ins.charging_rates[c] * sol.x[(k, c, t)]
                                                              for k in ins.customers_at_time[t] for c in
                                                              ins.customer_charging_options[k]])
        # round y then all its x's
        # calculated energy user of each customer
        for k in customers:
            energy_usage[k] = 0
            energy_usage_frac_sol[k] = 0
            # rounding y
            is_y_rounded = False
            if ins.rounding_tolerance < sol.y[k] < 1 - ins.rounding_tolerance:
                sol.y[k] = 0
                sol.frac_y_comp_count += 1
                is_y_rounded = True
            for t in ins.customer_charging_time_path[k]:
                for c in ins.customer_charging_options[k]:
                    energy_usage_frac_sol[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length
                    if is_y_rounded:
                        sol.x[(k, c, t)] = 0
                    energy_usage[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length

        #### Greedy rounding
        total_ev_charge_at_time = sol.total_ev_charge_at_time_frac_sol.copy()
        for t in np.arange(ins.scheduling_horizon):
            max_chg_option = {}
            for k in ins.customers_at_time[t]:
                options = [sol.x[k, cc, t] for cc in ins.customer_charging_options[k]]
                max_idx = np.argmax(options)
                max_chg_option[k] = ins.customer_charging_options[k][max_idx]
            # sort customers max x
            cust_x = {k: sol.x[(k, max_chg_option[k], t)] for k in ins.customers_at_time[t]}
            sorted_customers = [k for k, v in sorted(cust_x.iteritems(), key=lambda (k, v): (v, k), reverse=True) if v not in [0,1]]
            # print 'customers at time ',t, ins.customers_at_time[t]
            # print 'sorted customers at time ',t,sorted(cust_x.iteritems(), key=lambda (k, v): (v, k), reverse=True)
            # print 'sorted', sorted_customers

            for k in sorted_customers:
                if ins.rounding_tolerance < sol.x[(k, max_chg_option[k], t)] < 1 - ins.rounding_tolerance:
                    sol.frac_x_comp_count += 1

                    other_chg_options = np.setdiff1d(ins.customer_charging_options[k], [max_chg_option[k]])
                    rem_option_power = np.sum([sol.x[(k, c, t)] * ins.charging_rates[c] for c in
                                               other_chg_options])
                    total_power_after_rounding = total_ev_charge_at_time[t] + (1 - sol.x[k, max_chg_option[k], t]) * \
                                                                              ins.charging_rates[
                                                                                  max_chg_option[
                                                                                      k]] - rem_option_power

                    usage_after_rounding_down = energy_usage[k] - sol.x[k, max_chg_option[k], t] * ins.charging_rates[
                        max_chg_option[k]] * ins.step_length
                    if total_power_after_rounding <= (
                                ins.capacity_over_time[t] - ins.base_load_over_time[t]) and usage_after_rounding_down < \
                            ins.customer_usage[k]:
                        sol.x[k, max_chg_option[k], t] = 1
                    else:
                        sol.x[k, max_chg_option[k], t] = 0
                        energy_usage[k] = usage_after_rounding_down
                        total_power_after_rounding = total_ev_charge_at_time[t] - sol.x[k, max_chg_option[k], t] * \
                                                                                  ins.charging_rates[
                                                                                      max_chg_option[
                                                                                          k]] - rem_option_power
                    for c in other_chg_options:
                        sol.x[(k, c, t)] = 0

                    total_ev_charge_at_time[t] = total_power_after_rounding

        sol.count_y_due_to_rounded_x = 0
        sol.customer_satisfy_ratio = {}
        for k in customers:
            energy_usage[k] = 0
            for t in ins.customer_charging_time_path[k]:
                for c in ins.customer_charging_options[k]:
                    energy_usage[k] += sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length

        for k in customers:
            # set y variable
            sol.customer_satisfy_ratio[k] = energy_usage[k]/ins.customer_usage[k]
            if energy_usage[k] < ins.customer_usage[k] - ins.rounding_tolerance:
                sol.y[k] = 0
                sol.count_y_due_to_rounded_x += 1
                for c in ins.customer_charging_options[k]:
                    for t in ins.customer_charging_time_path[k]:
                        energy_usage[k] -= sol.x[k, c, t] * ins.charging_rates[c] * ins.step_length
                        sol.x[(k, c, t)] = 0

        sol.running_time = time.time() - t1

        # calculate new obj
        obj = 0
        for k in np.arange(ins.n):
            obj += ins.customer_utilities[k] * sol.y[k]
            for c in ins.customer_charging_options[k]:
                for t in ins.customer_charging_time_path[k]:
                    obj -= sol.x[(k, c, t)] * ins.cost_rate_matrix[c, t]
        # calculate P
        P = {}
        for t in np.arange(ins.scheduling_horizon):
            P[t] = ins.base_load_over_time[t]
            for k in ins.customers_at_time[t]:
                for c in ins.customer_charging_options[k]:
                    P[t] += sol.x[(k, c, t)] * ins.charging_rates[c]

        sol.frac_obj = sol.obj
        sol.obj = obj
        sol.customer_energy_usage = energy_usage
        sol.customer_energy_usage_frac_sol = energy_usage_frac_sol
        sol.P = P
        sol.ar = sol.obj/sol.frac_obj
    sol.running_time = time.time() - t1
    return sol

def round_EV_scheduling_fixed_interval(ins, guess_x={}, round_x_after_y=True):

    t1 = time.time()
    sol = max_ev_scheduling_OPT_fixed_interval(ins, guess_x=guess_x, fractional=True)
    frac_sol = copy.deepcopy(sol)
    sol.frac_sol = frac_sol

    sol.frac_x_comp_count = 0
    if sol.succeed:
        # customers = np.setdiff1d(np.arange(ins.n), guess_x)
        customers = np.arange(ins.n)
        sol.total_ev_charge_at_time_frac_sol = {}

        obj = 0
        for k in customers:
            for c in ins.customer_charging_options[k]:
                # rounding y
                if ins.rounding_tolerance < sol.x[k,c] < 1 - ins.rounding_tolerance:
                    sol.x[k,c]= 0
                    sol.frac_x_comp_count += 1


                    # calculate new obj
                obj += ins.customer_utilities[k,c] * sol.x[k,c]


        sol.running_time = time.time() - t1
        # calculate P
        P = {}
        for t in np.arange(ins.scheduling_horizon):
            P[t] = ins.base_load_over_time[t]
            for (k,c) in ins.customers_at_time[t]:
                P[t] += sol.x[(k, c)] * ins.charging_rates[c]

        sol.frac_obj = sol.obj
        sol.obj = obj
        sol.P = P
        sol.ar = sol.obj/sol.frac_obj
    sol.running_time = time.time() - t1
    return sol

# for e-energy 2018
def max_ev_scheduling_OPT_fixed_interval(ins, guess_x={}, fractional=False, debug=False, tolerance=0.001, alg="min_OPF_OPT"):
    t1 = time.time()
    m = gbp.Model("max_ev_scheduling_OPT_fixed_interval")
    u.gurobi_setting(m)
    x = {}  # key (i, c, t)

    P = {}
    # Q = {}


    time_path = np.arange(ins.scheduling_horizon)
    num_of_constraints = 0
    num_of_constraints_paper_formulation = 0

    for k in np.arange(ins.n):
        for c in ins.customer_charging_options[k]:
            if fractional:
                x[(k, c)] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[(%d,%d)]" % (k, c))
            else:
                x[(k, c)] = m.addVar(vtype=gbp.GRB.BINARY, name="x[(%d,%d)]" % (k, c))

    for t in time_path:
        P[t] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(t))

    obj = gbp.quicksum(x[k,c] * ins.customer_utilities[k,c] for k in range(ins.n) for c in ins.customer_charging_options[k])
    m.setObjective(obj, gbp.GRB.MAXIMIZE)
    m.update()

    for t in time_path:
        # capacity constraints
        ev_charge_list = [x[(k, c)] * (ins.charging_rates[c]) for (k,c) in ins.customers_at_time[t] if len(ins.customers_at_time[t]) != 0]
        rhs_P = ins.base_load_over_time[t] + gbp.quicksum(ev_charge_list)
        m.addConstr(P[t], gbp.GRB.EQUAL, rhs_P, "P_%d" % t)
        m.addConstr(P[t], gbp.GRB.LESS_EQUAL, ins.capacity_over_time[t], "C_%d" % t)
        num_of_constraints += 2
        num_of_constraints_paper_formulation += 1

    for k in np.arange(ins.n):
        # single charging option per customer contraints
        X = gbp.quicksum([x[(k, c)] for c in ins.customer_charging_options[k]])
        m.addConstr(X, gbp.GRB.LESS_EQUAL, 1, "sum_c X(%d) <= 1" % (k))
        num_of_constraints += 1
        num_of_constraints_paper_formulation += 1

        # box constraints
        for c in ins.customer_charging_options[k]:
            m.addConstr(x[(k, c)] >= 0, "x[%s]: lb" % str((k, c)))
            num_of_constraints += 1
            num_of_constraints_paper_formulation += 1

    # for k in guess_x.keys():
    #     m.addConstr(x[k], gbp.GRB.EQUAL, guess_x[k], "x[%d]: guess " % k)

    m.update()
    # m.write('model.lp')
    m.optimize()

    sol = a.OPF_EV_sol()
    sol.running_time = time.time() - t1
    # sol.gurobi_model = m

    if u.gurobi_handle_errors(m, algname=alg):
        sol.x = {(k, c): x[(k, c)].x for k in range(ins.n) for c in ins.customer_charging_options[k]}
        sol.obj = obj.getValue()
        sol.P = {t: P[t].x for t in np.arange(ins.scheduling_horizon)}
        sol.number_of_constraints_paper_formulation = num_of_constraints_paper_formulation
        sol.number_of_constraints = num_of_constraints
        sol.succeed = True
    else:
        sol.succeed = False

    return sol


# for e-energy 2018
def max_ev_scheduling_OPT(ins, guess_x={}, fractional=False, debug=False, tolerance=0.001, alg="min_OPF_OPT"):
    t1 = time.time()
    m = gbp.Model("max_ev_scheduling_OPT")
    u.gurobi_setting(m)
    x = {}  # key (i, c, t)
    y = {}

    P = {}
    # Q = {}

    customer_costs = []
    total_charge_per_customer = {}

    time_path = np.arange(ins.scheduling_horizon)
    num_of_constraints = 0
    num_of_constraints_paper_formulation = 0

    for k in np.arange(ins.n):
        if fractional:
            y[k] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="y[%d]" % k)
        else:
            y[k] = m.addVar(vtype=gbp.GRB.BINARY, name="y[%d]" % k)
        total_charge_per_customer[k] = 0
        for c in ins.customer_charging_options[k]:
            for t in ins.customer_charging_time_path[k]:
                if fractional:
                    x[(k, c, t)] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="x[(%d,%d,%d)]" % (k, c, t))
                else:
                    x[(k, c, t)] = m.addVar(vtype=gbp.GRB.BINARY, name="x[(%d,%d,%d)]" % (k, c, t))
                customer_costs.append(x[(k, c, t)] * ins.cost_rate_matrix[c, t])
                total_charge_per_customer[k] += x[k, c, t] * ins.charging_rates[c] * ins.step_length

    for t in time_path:
        P[t] = m.addVar(vtype=gbp.GRB.CONTINUOUS, name="P_%s" % str(t))

    obj = gbp.quicksum(y[k] * ins.customer_utilities[k] for k in range(ins.n)) - gbp.quicksum(customer_costs)
    m.setObjective(obj, gbp.GRB.MAXIMIZE)
    m.update()

    for t in time_path:
        # capacity constraints
        ev_charge_list = [x[(k, c, t)] * (ins.charging_rates[c]) for k in ins.customers_at_time[t] for
                          c in
                          ins.customer_charging_options[k] if len(ins.customers_at_time[t]) != 0]
        rhs_P = ins.base_load_over_time[t] + gbp.quicksum(ev_charge_list)
        m.addConstr(P[t], gbp.GRB.EQUAL, rhs_P, "P_%d" % t)
        m.addConstr(P[t], gbp.GRB.LESS_EQUAL, ins.capacity_over_time[t], "C_%d" % t)
        num_of_constraints += 2
        num_of_constraints_paper_formulation += 1

    for k in np.arange(ins.n):

        # single charging option per customer contraints
        for t in ins.customer_charging_time_path[k]:
            X = gbp.quicksum([x[(k, c, t)] for c in ins.customer_charging_options[k]])
            m.addConstr(X, gbp.GRB.LESS_EQUAL, 1, "sum_c X(%d,%s) <= 1" % (k, str(t)))
            num_of_constraints += 1
            num_of_constraints_paper_formulation += 1

        # usage constraint
        lhs = y[k] * ins.customer_usage[k] - total_charge_per_customer[k]
        m.addConstr(lhs, gbp.GRB.LESS_EQUAL, 0, "usage[%d]" % k)
        num_of_constraints += 1
        num_of_constraints_paper_formulation += 1

        # box constraints
        m.addConstr(y[k] <= 1, "y[%d]: ub")
        num_of_constraints += 1
        num_of_constraints_paper_formulation += 1
        for c in ins.customer_charging_options[k]:
            for t in ins.customer_charging_time_path[k]:
                m.addConstr(x[(k, c, t)] >= 0, "x[%s]: lb" % str((k, c, t)))
                num_of_constraints += 1
                num_of_constraints_paper_formulation += 1

    # for k in guess_x.keys():
    #     m.addConstr(x[k], gbp.GRB.EQUAL, guess_x[k], "x[%d]: guess " % k)

    m.update()
    # m.write('model.lp')
    m.optimize()

    sol = a.OPF_EV_sol()
    sol.running_time = time.time() - t1
    # sol.gurobi_model = m

    if u.gurobi_handle_errors(m, algname=alg):
        sol.x = {(k, c, t): x[(k, c, t)].x for k in range(ins.n) for t in ins.customer_charging_time_path[k]
                 for c in ins.customer_charging_options[k]}
        sol.y = {k: y[k].x for k in range(ins.n)}
        sol.total_charge_per_customer = total_charge_per_customer
        sol.obj = obj.getValue()
        sol.P = {t: P[t].x for t in np.arange(ins.scheduling_horizon)}
        sol.number_of_constraints_paper_formulation = num_of_constraints_paper_formulation
        sol.number_of_constraints = num_of_constraints
        sol.succeed = True
    else:
        sol.succeed = False

    return sol


# for FnT 2017
# No l terms and no voltage
def max_sOPF_OPT(ins, guess_x={}, fractional=False, debug=False, tolerance=0.001, alg="min_OPF_OPT"):
    assert (ins.cons == '' or ins.cons == 'V' or ins.cons == 'C')
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
    # m.write('model.lp')
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
def min_OPF_OPT(ins, guess_x={}, fractional=False, debug=False, tolerance=0.001, alg="min_OPF_OPT"):
    assert (ins.cons == '' or ins.cons == 'V' or ins.cons == 'C')
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
    obj = 1 + gbp.quicksum((1 - x[k]) * ins.loads_utilities[k] for k in range(ins.n)) + \
          T.graph['S_base'] * gbp.quicksum([l[e] * (T[e[0]][e[1]]['z'][0]) for e in T.edges()])
    m.setObjective(obj, gbp.GRB.MINIMIZE)

    for e in T.edges():
        z = T[e[0]][e[1]]['z']
        subtree_edges = nx.bfs_edges(T, e[1])

        m.addQConstr(l[e] * v[e[0]], gbp.GRB.GREATER_EQUAL, (P[e] * P[e] + Q[e] * Q[e]),
                     "l_%s" % str(e))  # l= |S|^2/ v_i

        # rhs_P = l[e] * z[0] + gbp.quicksum([x[i] * ins.loads_P[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [P[(e[1], h)] for h in T.edge[e[1]].keys()])  # + dummy_p[e[1]]
        rhs_P = l[e] * z[0] + gbp.quicksum([x[k] * ins.loads_P[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h] * z[0] for h in subtree_edges])

        m.addConstr(P[e], gbp.GRB.EQUAL, rhs_P, "P_%s" % str(e))
        # m.addConstr(P[e], gbp.GRB.LESS_EQUAL, rhs_P+tolerance, "P_%s" % str(e))
        # m.addConstr(P[e], gbp.GRB.GREATER_EQUAL, rhs_P-tolerance, "P_%s" % str(e))

        # rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
        #     [Q[(e[1], h)] for h in T.edge[e[1]].keys()]) # + dummy_q[e[1]]
        rhs_Q = l[e] * z[1] + gbp.quicksum([x[k] * ins.loads_Q[k] for k in T[e[0]][e[1]]['K']]) + gbp.quicksum(
            [l[h] * z[1] for h in subtree_edges])

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

        m.addConstr(v[e[1]] >= 0, "v_%d+:" % e[1])
        m.addConstr(l[e] >= 0, "l_%s+:" % str(e))



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
    # m.write('model.lp')
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


def min_OPF_OPT_erroneous(ins, guess_x={}, fractional=False, debug=False, tolerance=0.001, alg="min_OPF_OPT"):
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
        m.addConstr(P[e], gbp.GRB.GREATER_EQUAL, rhs_P - tolerance, "P_%s=" % str(e))
        m.addConstr(P[e], gbp.GRB.LESS_EQUAL, rhs_P + tolerance, "P_%s=" % str(e))

        rhs_Q = l[e] * z[1] + gbp.quicksum([x[i] * ins.loads_Q[i] for i in T.node[e[1]]['N']]) + gbp.quicksum(
            [Q[(e[1], h)] for h in T.edge[e[1]].keys() if len(T.edge[e[1]]) > 1])  # + dummy_q[e[1]]
        # m.addConstr(Q[e], gbp.GRB.EQUAL, rhs_Q, "Q_%s=" % str(e))
        m.addConstr(Q[e], gbp.GRB.GREATER_EQUAL, rhs_Q - tolerance, "Q_%s=" % str(e))
        m.addConstr(Q[e], gbp.GRB.LESS_EQUAL, rhs_Q + tolerance, "Q_%s=" % str(e))

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
