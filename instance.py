#! /usr/bin/python
__author__ = 'Majid Khonji'
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
import csv


# import util as u

############## IMPORTANT ###########
# index 0 is booked for the root
# nodes numbering follows breadth first order
####################################
class OPF_instance(object):
    def __init__(self):
        self.topology = None
        self.leaf_edges = {}  # leaf edges
        self.v_0 = 0
        self.v_max = 0
        self.v_min = 0
        self.V_ = 0  # (v_0 - v_min )/ 2
        self.n = 0  # number of customers
        self.Q = {}  # Q value of each customer. Dictionary keyed by (k,i) where k is customer and i is leaf vertex
        self.customer_path = {}  # path of customer k
        self.customer_leaf_path = {}  # path of customer k
        self.customer_node = {}  # customer to node
        self.load_theta_range = (0, 0)
        self.loads_S = None
        self.loads_P = None
        self.loads_Q = None
        self.loads_angles = None
        self.loads_utilities = None # or penalties
        self.I = []  # idx of integer customers
        self.F = []  # idx of fractional customers
        self.customer_path_nodes = {}  # path of customer k
        self.customer_leaf_path_nodes = {}  # path of customer k
        self.leaf_nodes = {}  # leaf nodes
        self.cons = ''
        self.gen_cost = 1
        ### some settings (maybe we should use a different structure in future)
        self.rounding_tolerance = 0.00001
        self.infeas_tolerance = 0.00001


class OPF_sol(object):
    def __init__(self):
        self.obj = 0;
        self.idx = [];  # if all are integer (F=[])
        self.running_time = 0;
        ar = None  # aproximation ratio
        self.topology = None  # updated graph with voltage, current,..etc filled
        self.x = {}  # used for fractional and mixed solutions
        self.P = {}
        self.Q = {}
        self.l = {}
        self.v = {}
        self.P_0 = 0 #real power from the root
        self.frac_comp_count = 0
        self.succeed = False
        self.gurobi_model= None


# cons=''|'V'|'C'|
# gen_cost cost of generation
def sim_instance(T, scenario="FCM", F_percentage=0.0, load_theta_range=(0, 1.2566370614359172),
                 n=10,
                 cus_load_range=(500, 5000), capacity_flag='C_',
                 ind_load_range=(300000, 1000000), industrial_cus_max_percentage=.2, v_0=1, v_max=1.21,
                 v_min=0.81000, cons='', gen_cost=.01):
    """
    Generates a random instance for CKP according to ckp_sim requirements
    """

    assert (scenario[0] in ['F', 'A'] and scenario[1] in ['C', 'U', '1'] and scenario[2] in ['R', 'I', 'M'])
    # initialize
    for k in T.nodes():
        T.node[k]['N'] = []  # customers on node i
        # T.node[k]['v'] = 0   # voltage
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers who's demands pass through edge e

    ins = OPF_instance()
    ins.cons = cons
    ins.gen_cost = gen_cost

    T.node[0]['v'] = ins.v_0
    ins.topology = T
    ins.n = n
    ins.leaf_nodes = T.graph['leaf_nodes']
    ins.leaf_edges = T.graph['leaf_edges']
    ins.F = np.random.choice(n, int(round(F_percentage * n)), replace=False)
    ins.I = np.setdiff1d(np.arange(n), ins.F)
    ins.v_0 = v_0
    ins.v_max = v_max
    ins.v_min = v_min
    ins.V_ = (v_0 - v_min) / 2
    loads_S = None
    loads_angles = None
    utilities = None

    r = 0  # index at which non industrial customer start until n

    if scenario[2] == 'R':  # commercial customers
        loads_S = np.random.uniform(cus_load_range[0], cus_load_range[1], n)
    elif scenario[2] == 'I':  # industrial
        loads_S = np.random.uniform(0, ind_load_range[1], n)
    elif scenario[2] == 'M':  # mixed
        r = np.random.randint(0, round(industrial_cus_max_percentage * n))
        industrial_loads = np.random.uniform(ind_load_range[0], ind_load_range[1], r)
        customer_loads = np.random.uniform(cus_load_range[0], cus_load_range[1], n - r)
        loads_S = np.append(industrial_loads, customer_loads)

    if scenario[0] == 'A':  # active  only
        loads_angles = np.zeros(n)
    elif scenario[0] == 'F':  # full: active and reactive
        loads_angles = np.random.uniform(load_theta_range[0], load_theta_range[1], n) * load_theta_range[0]

    if scenario[1] == 'C':  # correlated demand/utility
        utilities = loads_S ** 2
    elif scenario[1] == 'U':  # uncorrelated
        util_func = lambda x: random.randrange(0, x)
        utilities = np.zeros(n)
        utilities[0:r] = np.array([util_func(ind_load_range[1]) for i in range(r)])
        utilities[r:n] = np.array([util_func(cus_load_range[1]) for i in range(n - r)])

    ins.loads_S = loads_S / T.graph['S_base']
    ins.loads_angles = loads_angles
    ins.loads_utilities = utilities
    ins.loads_P = np.array(map(lambda x, t: x * np.math.cos(t), ins.loads_S, ins.loads_angles))
    ins.loads_Q = np.array(map(lambda x, t: x * np.math.sin(t), ins.loads_S, ins.loads_angles))

    # print "angles ", loads_angles
    # print "util  ", utilities
    _distribute_customers(ins, capacity_flag=capacity_flag)

    return ins


# returns T  of type nx.Graph()
# loss C_ or L flags for T are not set
def network_csv_load(filename='test-feeders/123-node-line-data.csv', S_base=5000000, V_base=4160, visualize=False,
                     loss_ratio=0.0):
    # T = nx.Graph()
    T = nx.DiGraph()
    T.graph["S_base"] = S_base
    T.graph["V_base"] = V_base
    with open(filename) as csvfile:
        reader = csv.DictReader(filter(lambda row: row[0] != '#', csvfile));
        for row in reader:
            # print row['Node A'], "|" , row['Node B'], "|", row['r (p.u.)'], '|' , row['x (p.u.)'], '|', row['C (p.u.)']
            A = int(row['Node A'])
            B = int(row['Node B'])
            T.add_edge(A, B)
            T[A][B]['z'] = (float(row['r (p.u.)']), float(row['x (p.u.)']))
            T[A][B]['C'] = float(row['C (p.u.)'])

    # print T.edges()
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers who's demands pass through edge e
    nx.set_edge_attributes(T, 'L', {k: 0 for k in T.edges()})  # loss upper bound
    nx.set_node_attributes(T, 'depth', {k: 0 for k in T.nodes()})
    nx.set_node_attributes(T, 'path', {k: nx.shortest_path(T,0,k) for k in T.nodes()})
    leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    T.graph['leaf_nodes'] = leaf_nodes
    T.graph['leaf_edges'] = [(i, nx.predecessor(T, 0, i)[0]) for i in leaf_nodes]
    max_depth = 0
    for i in T.nodes()[1:]:
        T.node[i]['depth'] = len(nx.shortest_path(T, 0, i)) - 1
        if T.node[i]['depth'] > max_depth: max_depth = T.node[i]['depth']
    logging.info('max depth  = %d' % max_depth)

    # setting loss to fixed
    for e in T.edges():
        C = T[e[0]][e[1]]['C']
        z = T[e[0]][e[1]]['z']
        T[e[0]][e[1]]['L'] = C * loss_ratio
        T[e[0]][e[1]]['C_'] = C * (1 - loss_ratio)
        logging.info("=== edge %s ===" % str(e))
        logging.info('C     = %.4f , |z| = %.6f' % (C, np.sqrt(z[0] ** 2 + z[1] ** 2)))
        logging.info('loss  = %.6f\t %4.2f%%' % (T[e[0]][e[1]]['L'], T[e[0]][e[1]]['L'] / C * 100))
    if visualize:
        pos = nx.graphviz_layout(T, prog='dot')
        nx.draw(T, pos)
        node_labels = {k: k for k in T.nodes()}
        nx.draw_networkx_labels(T, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(T, 'z')
        nx.draw_networkx_edge_labels(T, pos, labels=edge_labels)
        plt.show()

    return T


# method = ['method1'|'method2'|'fixed'] (experimental)
# loss C_ or L flags for T are used later.
def network_38node(visualize=False, loss_ratio=.08,
                   method='fixed'):
    T = nx.DiGraph()
    T.graph["S_base"] = 1000000
    T.graph["V_base"] = 12660
    T.add_edge(0, 2);
    T[0][2]['z'] = (.000574, .000293);
    T[0][2]['C'] = 4.6
    T.add_edge(2, 3);
    T[2][3]['z'] = (.00307, .001564);
    T[2][3]['C'] = 4.1
    T.add_edge(3, 4);
    T[3][4]['z'] = (.002279, .001161);
    T[3][4]['C'] = 2.9
    T.add_edge(4, 5);
    T[4][5]['z'] = (.002373, .001209);
    T[4][5]['C'] = 2.9
    T.add_edge(5, 6);
    T[5][6]['z'] = (.0051, .004402);
    T[5][6]['C'] = 2.9
    T.add_edge(6, 7);
    T[6][7]['z'] = (.001166, .003853);
    T[6][7]['C'] = 1.5
    T.add_edge(7, 8);
    T[7][8]['z'] = (.00443, .001464);
    T[7][8]['C'] = 1.05
    T.add_edge(8, 9);
    T[8][9]['z'] = (.006413, .004608);
    T[8][9]['C'] = 1.05
    T.add_edge(9, 10);
    T[9][10]['z'] = (.006501, .004608);
    T[9][10]['C'] = 1.05
    T.add_edge(10, 11);
    T[10][11]['z'] = (.001224, .000405);
    T[10][11]['C'] = 1.05
    T.add_edge(11, 12);
    T[11][12]['z'] = (.002331, .000771);
    T[11][12]['C'] = 1.05
    T.add_edge(12, 13);
    T[12][13]['z'] = (.009141, .007192);
    T[12][13]['C'] = 0.5
    T.add_edge(13, 14);
    T[13][14]['z'] = (.003372, .004439);
    T[13][14]['C'] = 0.45
    T.add_edge(14, 15);
    T[14][15]['z'] = (.00368, .003275);
    T[14][15]['C'] = 0.3
    T.add_edge(15, 16);
    T[15][16]['z'] = (.004647, .003394);
    T[15][16]['C'] = 0.25
    T.add_edge(16, 17);
    T[16][17]['z'] = (.008026, .010716);
    T[16][17]['C'] = 0.25
    T.add_edge(17, 18);
    T[17][18]['z'] = (.004558, .003574);
    T[17][18]['C'] = 0.1
    T.add_edge(2, 19);
    T[2][19]['z'] = (.001021, .000974);
    T[2][19]['C'] = 0.5
    T.add_edge(19, 20);
    T[19][20]['z'] = (.009366, .00844);
    T[19][20]['C'] = 0.5
    T.add_edge(20, 21);
    T[20][21]['z'] = (.00255, .002979);
    T[20][21]['C'] = 0.21
    T.add_edge(21, 22);
    T[21][22]['z'] = (.004414, .005836);
    T[21][22]['C'] = 0.11
    T.add_edge(3, 23);
    T[3][23]['z'] = (.002809, .00192);
    T[3][23]['C'] = 1.05
    T.add_edge(23, 24);
    T[23][24]['z'] = (.005592, .004415);
    T[23][24]['C'] = 1.05
    T.add_edge(24, 25);
    T[24][25]['z'] = (.005579, .004366);
    T[24][25]['C'] = 0.5
    T.add_edge(6, 26);
    T[6][26]['z'] = (.001264, .000644);
    T[6][26]['C'] = 1.5
    T.add_edge(26, 27);
    T[26][27]['z'] = (.00177, .000901);
    T[26][27]['C'] = 1.5
    T.add_edge(27, 28);
    T[27][28]['z'] = (.006594, .005814);
    T[27][28]['C'] = 1.5
    T.add_edge(28, 29);
    T[28][29]['z'] = (.005007, .004362);
    T[28][29]['C'] = 1.5
    T.add_edge(29, 30);
    T[29][30]['z'] = (.00316, .00161);
    T[29][30]['C'] = 1.5
    T.add_edge(30, 31);
    T[30][31]['z'] = (.006067, .005996);
    T[30][31]['C'] = 0.5
    T.add_edge(31, 32);
    T[31][32]['z'] = (.001933, .002253);
    T[31][32]['C'] = 0.5
    T.add_edge(32, 33);
    T[32][33]['z'] = (.002123, .003301);
    T[32][33]['C'] = 0.1
    T.add_edge(8, 34);
    T[8][34]['z'] = (.012453, .012453);
    T[8][34]['C'] = 0.5
    T.add_edge(9, 35);
    T[9][35]['z'] = (.012453, .012453);
    T[9][35]['C'] = 0.5
    T.add_edge(12, 36);
    T[12][36]['z'] = (.012453, .012453);
    T[12][36]['C'] = 0.5
    T.add_edge(18, 37);
    T[18][37]['z'] = (.003113, .003113);
    T[18][37]['C'] = 0.5
    T.add_edge(25, 38);
    T[25][38]['z'] = (.003113, .003113);
    T[25][38]['C'] = 0.1

    # max_z_theta = 0
    # for e in T.edges():
    #     re = T[e[0]][e[1]]['z'][0]
    #     im = T[e[0]][e[1]]['z'][1]
    #     ang = np.arctan(im/re)
    #     print 'ang of %s is '%str(e), ang*180/np.pi
    #     if ang > max_z_theta: max_z_theta = ang
    # print 'max load angle ', max_z_theta * 180/np.pi


    # nx.set_edge_attributes(T, 'l', {k: 0 for k in T.edges()})  # current magnitude square
    # nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    # nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers whos demands pass through edge e
    # nx.set_node_attributes(T, 'v', {k: 0 for k in T.nodes()})  # voltage magnitude square
    nx.set_edge_attributes(T, 'L', {k: 0 for k in T.edges()})  # loss upper bound
    nx.set_node_attributes(T, 'depth', {k: 0 for k in T.nodes()})
    leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    T.graph['leaf_nodes'] = leaf_nodes
    T.graph['leaf_edges'] = [(i, nx.predecessor(T, 0, i)[0]) for i in leaf_nodes]
    max_depth = 0
    for i in T.nodes()[1:]:
        T.node[i]['depth'] = len(nx.shortest_path(T, 0, i)) - 1
        if T.node[i]['depth'] > max_depth: max_depth = T.node[i]['depth']
    logging.info('max depth  = %d' % max_depth)

    ####### method 1: pessimistic (experimental)
    if method == "method1":
        max_loss_per = 0
        for e in T.edges():
            print "=== edge %s ===" % str(e)
            C = T[e[0]][e[1]]['C']
            z = T[e[0]][e[1]]['z']
            print 'C   = %.4f , |z| = %.6f' % (C, np.sqrt(z[0] ** 2 + z[1] ** 2))
            if e[0] == 0:  # first edge
                # T[e[0]][e[1]]['L'] = np.sqrt(z[0]**2 + z[1]**2)*  C**2
                T[e[0]][e[1]]['L'] = (max_depth - T.node[e[1]]['depth']) * np.sqrt(z[0] ** 2 + z[1] ** 2) * C ** 2
            else:
                T[e[0]][e[1]]['L'] = (max_depth - T.node[e[1]]['depth']) * np.sqrt(z[0] ** 2 + z[1] ** 2) * C ** 2
                # T[e[0]][e[1]]['L'] =  np.sqrt(z[0]**2 + z[1]**2)*  C**2
            per = T[e[0]][e[1]]['L'] / C * 100
            if per > max_loss_per: max_loss_per = per
            print 'loss  = %.6f\t %4.2f%%' % (T[e[0]][e[1]]['L'], per)
            # print 'max %% = %.2f'%max_loss_per
    elif method == "method2":
        ####### method 2: ###### (experimental)
        print "##### method 2 #####"
        for e in T.edges():
            T_ = T.copy()
            if e[0] != 0:  # not root, then delete the dege connected to e[0]
                # disconnect the graph at e[0]
                parent = nx.predecessor(T, 0, e[0])[0]
                T_.remove_node(parent)
                T_ = nx.bfs_tree(T_, e[0])
            C = T[e[0]][e[1]]['C']
            z_max = {d: [] for d in range(1, max_depth - T.node[e[0]]['depth'] + 1)}
            for i in T_.nodes():
                if i != e[0]:
                    parent = nx.predecessor(T_, e[0], i)[0]
                    # print "(%d,%d)"%(parent,i), " resistance is ", np.sqrt(T[parent][i]['z'][0]**2 + T[parent][i]['z'][1]**2)
                    depth = T.node[i]['depth'] - T.node[e[0]]['depth']
                    z_max[depth].append(np.sqrt(T[parent][i]['z'][0] ** 2 + T[parent][i]['z'][0] ** 2))
            z_max = {l: max(v) for l, v in z_max.iteritems() if v != []}
            loss1 = sum([C ** 2 * z for l, z in z_max.iteritems()])  # very bad
            loss2 = 0
            for d in z_max.keys():
                loss2 += z_max[d] * (C - loss2) ** 2
            # print "loss of %-8s  = %.6f"%(str(e), loss1)
            T[e[0]][e[1]]['L'] = loss2
            print "loss of %-8s  = %.6f\t %4.2f%%" % (str(e), loss2, loss2 / C)
    ######### default method #######
    elif method == "fixed":
        for e in T.edges():
            C = T[e[0]][e[1]]['C']
            z = T[e[0]][e[1]]['z']
            T[e[0]][e[1]]['L'] = C * loss_ratio
            T[e[0]][e[1]]['C_'] = C * (1 - loss_ratio)
            logging.info("=== edge %s ===" % str(e))
            logging.info('C     = %.4f , |z| = %.6f' % (C, np.sqrt(z[0] ** 2 + z[1] ** 2)))
            logging.info('loss  = %.6f\t %4.2f%%' % (T[e[0]][e[1]]['L'], T[e[0]][e[1]]['L'] / C * 100))
    if visualize:
        pos = nx.graphviz_layout(T, prog='dot')
        nx.draw(T, pos)
        node_labels = {k: k for k in T.nodes()}
        nx.draw_networkx_labels(T, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(T, 'z')
        nx.draw_networkx_edge_labels(T, pos, labels=edge_labels)
        plt.show()

    return T


def rnd_instance_from_graph(T, n=3, v_0=1, v_max=1.21, v_min=0.81000,
                            load_theta_range=(-0.628, 0.628), max_load=.1,
                            util_func=lambda x, y: x - x + np.random.rand()):
    # initialize
    for k in T.nodes():
        T.node[k]['N'] = []  # customers on node i
        # T.node[k]['v'] = 0   # voltage
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers who's demands pass through edge e

    ins = OPF_instance()
    T.node[0]['v'] = ins.v_0
    ins.topology = T
    ins.n = n
    ins.leaf_nodes = T.graph['leaf_nodes']
    ins.leaf_edges = T.graph['leaf_edges']
    ins.I = np.arange(n)
    ins.F = np.array([], dtype=np.int64)

    ins.v_0 = v_0
    ins.v_max = v_max
    ins.v_min = v_min
    ins.V_ = (v_0 - v_min) / 2
    ins.loads_S = np.random.rand(n) * max_load
    ins.loads_angles = np.random.uniform(load_theta_range[0], load_theta_range[1], n) * load_theta_range[0]
    ins.loads_utilities = np.array(map(util_func, ins.loads_S, ins.loads_angles))
    _distribute_customers(ins)
    return ins


# topology must be filled in instance ins
def _distribute_customers(ins, capacity_flag='C_'):
    T = ins.topology
    n = ins.n

    for k in range(n):
        attached_node = np.random.choice(T.nodes()[1:])
        parent = nx.predecessor(T, 0, attached_node)[0]
        while ins.loads_S[k] > T[parent][attached_node][capacity_flag]:
            attached_node = np.random.choice(T.nodes()[1:])
            parent = nx.predecessor(T, 0, attached_node)[0]

        T.node[attached_node]['N'].append(k)
        ins.customer_node[k] = attached_node
        ins.customer_path_nodes[k] = T.node[attached_node]['path']

        ins.customer_path[k] = zip(ins.customer_path_nodes[k], ins.customer_path_nodes[k][1:]) # edges
        for e in ins.customer_path[k]: T[e[0]][e[1]]['K'].append(k)

        #computingn Q values at leafs
        for l in ins.leaf_nodes:
            ins.customer_leaf_path[(k, l)] = []
            leaf_path = T.node[l]['path']

            min_depth = np.min([len(leaf_path), len(ins.customer_path_nodes[k])])
            ins.customer_leaf_path_nodes[(k, l)] = [leaf_path[i] for i in
                                                    range(min_depth) if
                                                    leaf_path[i] == ins.customer_path_nodes[k][i]]

            ins.customer_leaf_path[(k, l)] = zip(ins.customer_leaf_path_nodes[(k, l)],
                                                 ins.customer_leaf_path_nodes[(k, l)][1:])

            ins.Q[(k, l)] = 0
            for e in ins.customer_leaf_path[(k, l)]:
                ins.Q[(k, l)] += T[e[0]][e[1]]['z'][0] * ins.loads_P[k] + T[e[0]][e[1]]['z'][1] * ins.loads_Q[k]

# with debugging  options
def _slow_distribute_customers(ins, capacity_flag='C_'):
    T = ins.topology
    n = ins.n
    # ins.loads_P = np.array(map(lambda x, t: x * np.math.cos(t), ins.loads_S, ins.loads_angles))
    # ins.loads_Q = np.array(map(lambda x, t: x * np.math.sin(t), ins.loads_S, ins.loads_angles))

    for k in range(n):
        attached_node = np.random.choice(T.nodes()[1:])
        parent = nx.predecessor(T, 0, attached_node)[0]
        while ins.loads_S[k] > T[parent][attached_node][capacity_flag]:
            attached_node = np.random.choice(T.nodes()[1:])
            parent = nx.predecessor(T, 0, attached_node)[0]

        T.node[attached_node]['N'].append(k)
        ins.customer_node[k] = attached_node
        ins.customer_path_nodes[k] = nx.shortest_path(T, 0, attached_node)
        ins.customer_path[k] = zip(ins.customer_path_nodes[k], ins.customer_path_nodes[k][1:]) # edges
        for e in ins.customer_path[k]: T[e[0]][e[1]]['K'].append(k)

        #########
        logging.debug('------- customer: %d -------' % k)
        logging.debug('demand      : (%f,%f)' % (ins.loads_P[k], ins.loads_Q[k]))
        logging.debug('S           : %f' % (ins.loads_S[k]))
        logging.debug('attached bus: %d' % attached_node)
        logging.debug('path        : %s' % str(ins.customer_path_nodes[k]))
        #########
        for l in ins.leaf_nodes:
            logging.debug('---')
            ins.customer_leaf_path[(k, l)] = []
            leaf_path = nx.shortest_path(T, 0, l)
            logging.debug("leaf path: %s" % str(leaf_path))

            # ins.customer_leaf_path_nodes[(k, l)] = [i for i in range(np.min([len(leaf_path), len(ins.customer_path_nodes[k])]))
            #                                   if leaf_path[i] == ins.customer_path_nodes[k][i]]

            min_depth = np.min([len(leaf_path), len(ins.customer_path_nodes[k])])
            ins.customer_leaf_path_nodes[(k, l)] = [leaf_path[i] for i in
                                                    range(min_depth) if
                                                    leaf_path[i] == ins.customer_path_nodes[k][i]]
            ins.customer_leaf_path[(k, l)] = zip(ins.customer_leaf_path_nodes[(k, l)],
                                                 ins.customer_leaf_path_nodes[(k, l)][1:])

            logging.debug("shared path with leaf %d : %s" % (l, str(ins.customer_leaf_path_nodes[(k, l)])))
            logging.debug("shared edges with leaf %d: %s" % (l, str(ins.customer_leaf_path[(k, l)])))
            ins.Q[(k, l)] = 0
            tmp_str = "path impedance through leaf %d: " % l
            for e in ins.customer_leaf_path[(k, l)]:
                ins.Q[(k, l)] += T[e[0]][e[1]]['z'][0] * ins.loads_P[k] + T[e[0]][e[1]]['z'][1] * ins.loads_Q[k]
                tmp_str += '(%f, %f) ' % (T[e[0]][e[1]]['z'][0], T[e[0]][e[1]]['z'][1])
            logging.debug(tmp_str)
            logging.debug('Q[(%d,%d)]=%f' % (k, l, ins.Q[(k, l)]))

def single_link_instance(n=10, capacity=1, z=(.01, .01), loss_ratio=.08, S_base=5000000, Rand=True):
    ins = OPF_instance()
    ins.n = n
    ins.I = np.arange(n)
    ins.F = np.array([], dtype=np.int64)
    ins.v_0 = 1
    ins.v_max = 1.05
    ins.v_min = 0.81000
    ins.V_ = (ins.v_0 - ins.v_min) / 2
    T = nx.Graph()
    T.graph["S_base"] = S_base
    T.add_edge(0, 1)
    nx.set_edge_attributes(T, 'C', {k: capacity for k in T.edges()})  # max VA capacity
    nx.set_edge_attributes(T, 'C_', {k: capacity * (1 - loss_ratio) for k in T.edges()})  # max VA capacity minus loss
    nx.set_edge_attributes(T, 'l', {k: 0 for k in T.edges()})  # current magnitude square
    nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    nx.set_edge_attributes(T, 'z', {k: z for k in T.edges()})  # impedence [complex
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers whos demands pass through edge e
    nx.set_node_attributes(T, 'v', {k: 0 for k in T.nodes()})  # voltage magnitude square

    T.node[1]['N'] = ins.I  # customers on node i
    T.node[1]['v'] = 0  # voltage
    T.node[0]['v'] = ins.v_0
    ins.leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    ins.leaf_edges = [(i, nx.predecessor(T, i, 0)[0]) for i in ins.leaf_nodes]

    # ins.loads_P = np.array([0.06, 1.63, 1.91, 1.47, 1.12, 1.34, 1.02, 1.66, 1.34, 0.15])
    # ins.loads_Q = np.array([0]*n)
    # ins.loads_S = np.array([0.06, 1.63, 1.91, 1.47, 1.12, 1.34, 1.02, 1.66, 1.34, 0.15])
    # ins.loads_utilities = np.array([0.59, 0.58, 0.67, 0.3, 0.9, 0.3, 0.27, 0.79, 0.55, 0.7])
    if Rand:
        angle = 35 * np.pi / 180
        ins.loads_P = np.random.uniform(0, 1, ins.n)
        ins.loads_Q = np.array([r * np.sin(angle) * (-1) ** np.random.randint(0, 2) for r in ins.loads_P])
        ins.loads_S = np.array([np.sqrt(ins.loads_P[k] ** 2 + ins.loads_Q[k] ** 2) for k in range(ins.n)])
        ins.loads_utilities = [np.sqrt(ins.loads_P[k] ** 2 + ins.loads_Q[k] ** 2) for k in range(ins.n)]
    else:
        ins.loads_P = np.array([1] * n)
        ins.loads_Q = np.array([0] * n)
        ins.loads_S = np.array([1] * n)
        ins.loads_utilities = ins.loads_S

    ins.Q = {(k, 1): z[0] * ins.loads_P[k] + z[1] * ins.loads_Q[k] for k in range(n)}
    ins.customer_path = {k: [(0, 1)] for k in range(n)}

    T[0][1]['K'] = ins.I

    ins.topology = T
    return ins


def rnd_path_instance(n=3, node_count=3, v_0=1, v_max=1.05, v_min=.95, capacity_range=(5, 10), max_load_theta=0.628,
                      max_imp_theta=0.628, max_resistance=.1,
                      max_load=2,
                      util_func=lambda x, y: x - x + np.random.rand()):
    ins = OPF_instance()
    ins.I = np.arange(n)
    ins.v_0 = v_0
    ins.v_max = v_max
    ins.v_min = v_min
    ins.V_ = (v_0 - v_min) / 2

    T = nx.path_graph(node_count)
    nx.set_edge_attributes(T, 'C', {k: np.random.uniform(capacity_range[0], capacity_range[1]) for k in
                                    T.edges()})  # max VA capacity
    nx.set_edge_attributes(T, 'l', {k: 0 for k in T.edges()})  # current magnitude square
    nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    nx.set_edge_attributes(T, 'z',
                           {k: (max_resistance * np.cos(max_imp_theta), max_resistance * np.sin(max_imp_theta)) for k in
                            T.edges()})  # impedence [complex
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers whos demands pass through edge e
    for k in T.nodes():
        T.node[k]['N'] = []  # customers on node i
        T.node[k]['v'] = 0  # voltage
    T.node[0]['v'] = ins.v_0
    ins.leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    ins.leaf_edges = [(i, nx.predecessor(T, i, 0)[0]) for i in ins.leaf_nodes]

    ins.topology = T
    _distribute_customers(ins, n=n, load_theta_range=(0, max_load_theta), max_load=max_load, util_func=util_func)
    return ins


def rnd_tree_instance(n=3, depth=2, branch=2, v_0=1, v_max=1.21, v_min=0.81000, impedence_range=(.0008, 0.015),
                      capacity_range=(1, 4),
                      max_load_theta=0.628,
                      max_imp_theta=0.628,
                      max_load=2,
                      util_func=lambda x, y: x - x + np.random.rand()):
    ins = OPF_instance()
    ins.I = np.arange(n)
    ins.v_0 = v_0
    ins.v_max = v_max
    ins.v_min = v_min
    ins.V_ = (v_0 - v_min) / 2

    T = nx.balanced_tree(depth, branch)
    mapping = {k: k + 1 for k in T.nodes()}
    T = nx.relabel_nodes(T, mapping)
    T.add_edge(0, 1)
    # T.add_path(range(node_count))
    nx.set_edge_attributes(T, 'C', {k: np.random.uniform(capacity_range[0], capacity_range[1]) for k in
                                    T.edges()})  # max VA capacity
    nx.set_edge_attributes(T, 'z', {k: (.001, .001) for k in T.edges()})  # impedence [complex
    # nx.set_edge_attributes(T, 'z', {k: (np.random.uniform(impedence_range[0], impedence_range[1]), \
    #                                     np.random.uniform(impedence_range[0], impedence_range[1])) for k in T.edges()})  # impedence [complex
    nx.set_edge_attributes(T, 'l', {k: 0 for k in T.edges()})  # current magnitude square
    nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers whos demands pass through edge e
    nx.set_node_attributes(T, 'v', {k: 0 for k in T.nodes()})  # voltage magnitude square
    for k in T.nodes():
        T.node[k]['N'] = []  # customers on node i
        T.node[k]['v'] = 0  # voltage
    T.node[0]['v'] = ins.v_0

    ins.topology = T
    ins.leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    ins.leaf_edges = [(i, nx.predecessor(T, i, 0)[0]) for i in ins.leaf_nodes]
    _distribute_customers(ins, n=n, load_theta_range=(0, max_load_theta), max_load=max_load, util_func=util_func)
    return ins


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    # ins = rnd_tree_instance()
    # ins = single_link_instance()
    # print_instance(ins)
    # ins = rnd_path_instance()
    # logging.debug("Q: %s"%str(ins.Q) )
    # print ins.topology.edges()
    # print ins.topology.node[0]
    # plt.show()

    T = network_38node()
    # ins = rnd_instance_from_graph()
    ins = sim_instance(T, scenario="FUI", n=1, ind_load_range=(1000000, 4000000))
    # u.print_instance(ins)
