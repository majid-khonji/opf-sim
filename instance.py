#! /usr/bin/python
__author__ = 'majid'
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
import maxOPF_algs as m

############## IMPORTANT ###########
# index 0 is booked for the root
# nodes numbering follows breadth first order
####################################
class maxOPF_instance(object):
    topology = None
    leaf_edges = {}  # leaf edges
    v_0 = 0
    v_max = 0
    v_min = 0
    V_ = 0  # (v_0 - v_min )/ 2
    n = 0  # number of customers
    Q = {}  # Q value of each customer. Dictionary keyed by (k,i) where k is customer and i is leaf vertex
    customer_path = {}  # path of customer k
    customer_leaf_path = {}  # path of customer k
    customer_node = {}  # customer to node
    load_theta_range = (0,0)
    loads_S = None
    loads_P = None
    loads_Q = None
    loads_angles = None
    loads_utilities = None

    I = []  # idx of integer customers
    F = []  # idx of fractional customers
    customer_path_nodes = {}  # path of customer k
    customer_leaf_path_nodes = {}  # path of customer k
    leaf_nodes = {}  # leaf nodes


class maxOPF_sol(object):
    def __init__(self):
        self.obj = 0;
        self.idx = []; # if all are integer (F=[])
        self.running_time = 0;
        ar = None  # aproximation ratio
        self.topology = None  # updated graph with voltage, current,..etc filled
        self.x = [] # used fractional and mixed solutions
        self.P = {}
        self.Q = {}
        self.l = {}
        self.v = {}


def rnd_path_instance(n=3, node_count=3, v_0=1, v_max=1.05, v_min=.95, capacity_range=(5, 10), max_load_theta=0.628,
                      max_imp_theta=0.628, max_resistance = .1,
                      max_load=2,
                      util_func=lambda x, y: x - x + np.random.rand()):
    ins = maxOPF_instance()
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
    nx.set_edge_attributes(T, 'z', {k: (max_resistance * np.cos(max_imp_theta),  max_resistance * np.sin(max_imp_theta)) for k in T.edges()})  # impedence [complex
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers whos demands pass through edge e
    for k in T.nodes():
        T.node[k]['N'] = []  # customers on node i
        T.node[k]['v'] = 0  # voltage
    T.node[0]['v'] = ins.v_0
    ins.leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    ins.leaf_edges = [(i, nx.predecessor(T, i, 0)[0]) for i in ins.leaf_nodes]

    ins.topology = T
    set_customers(ins, n=n, load_theta_range=(0,max_load_theta), max_load=max_load, util_func=util_func)
    return ins


def rnd_tree_instance(n=3, depth=2, branch=2, v_0=1, v_max=1.21, v_min=0.81000, impedence_range=(.0008, 0.015), capacity_range=(1, 4),
                      max_load_theta=0.628,
                      max_imp_theta=0.628,
                      max_load=2,
                      util_func=lambda x, y: x - x + np.random.rand()):
    ins = maxOPF_instance()
    ins.I = np.arange(n)
    ins.v_0 = v_0
    ins.v_max = v_max
    ins.v_min = v_min
    ins.V_ = (v_0 - v_min) / 2

    T = nx.balanced_tree(depth, branch)
    mapping = {k: k + 1 for k in T.nodes()}
    T = nx.relabel_nodes(T, mapping)
    T.add_edge(0,1)
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
        T.node[k]['v'] = 0   # voltage
    T.node[0]['v'] = ins.v_0

    ins.topology = T
    set_customers(ins, n=n, load_theta_range=(0,max_load_theta), max_load=max_load, util_func=util_func)
    return ins


def single_link_instance(n=10, capacity=.1, z = (.01,.01),Rand = True):
    ins = maxOPF_instance()
    ins.n = n
    ins.I = np.arange(n)
    ins.v_0 = 1
    ins.v_max = 1.05
    ins.v_min = 0.81000
    ins.V_ = (ins.v_0 - ins.v_min) / 2
    T = nx.Graph()
    T.add_edge(0, 1)
    nx.set_edge_attributes(T, 'C', {k: capacity for k in
                                    T.edges()})  # max VA capacity
    nx.set_edge_attributes(T, 'l', {k: 0 for k in T.edges()})  # current magnitude square
    nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    nx.set_edge_attributes(T, 'z', {k: z for k in T.edges()})  # impedence [complex
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers whos demands pass through edge e
    nx.set_node_attributes(T, 'v', {k: 0 for k in T.nodes()})  # voltage magnitude square

    T.node[1]['N'] = ins.I  # customers on node i
    T.node[1]['v'] = 0  # voltage
    T.node[0]['v'] = ins.v_0
    # ins.leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    # ins.leaf_edges = [(i, nx.predecessor(T, i, 0)[0]) for i in ins.leaf_nodes]

    # ins.loads_P = np.array([0.06, 1.63, 1.91, 1.47, 1.12, 1.34, 1.02, 1.66, 1.34, 0.15])
    # ins.loads_Q = np.array([0]*n)
    # ins.loads_S = np.array([0.06, 1.63, 1.91, 1.47, 1.12, 1.34, 1.02, 1.66, 1.34, 0.15])
    # ins.loads_utilities = np.array([0.59, 0.58, 0.67, 0.3, 0.9, 0.3, 0.27, 0.79, 0.55, 0.7])
    if Rand:
        angle = 35 * np.pi/180
        ins.loads_P = np.random.uniform(0,1,ins.n)
        ins.loads_Q = np.array([r*np.sin(angle)*(-1)**np.random.randint(0,2) for r in ins.loads_P])
        ins.loads_S = np.array([np.sqrt(ins.loads_P[k]**2 + ins.loads_Q[k]**2) for k in range(ins.n)])
        ins.loads_utilities = [np.sqrt(ins.loads_P[k]**2 + ins.loads_Q[k]**2) for k in range(ins.n)]
    else:
        ins.loads_P = np.array([1]*n)
        ins.loads_Q = np.array([0]*n)
        ins.loads_S = np.array([1]*n)
        ins.loads_utilities = ins.loads_S

    ins.Q = {(k,1): z[0] * ins.loads_P[k] + z[1]*ins.loads_Q[k] for k in range(n)}
    ins.customer_path = {k: [(0,1)] for k in range(n)}

    T[0][1]['K'] = ins.I

    ins.topology = T
    return ins

# method = ['method1'|'method2'|'fixed']
def network_38node(file_name='data/38_node_test.gpickle', visualize = False,v_0=1, v_min=0.81000, loss_ratio = .08,
        method='fixed'):
    T = nx.Graph()
    T.add_edge(0,2); T[0][2]['z'] = (.000574, .000293); T[0][2]['C'] = 4.6
    T.add_edge(2,3); T[2][3]['z'] = (.00307, .001564); T[2][3]['C'] = 4.1
    T.add_edge(3,4); T[3][4]['z'] = (.002279, .001161); T[3][4]['C'] = 2.9
    T.add_edge(4,5); T[4][5]['z'] = (.002373, .001209); T[4][5]['C'] = 2.9
    T.add_edge(5,6); T[5][6]['z'] = (.0051, .004402); T[5][6]['C'] = 2.9
    T.add_edge(6,7); T[6][7]['z'] = (.001166, .003853); T[6][7]['C'] = 1.5
    T.add_edge(7,8); T[7][8]['z'] = (.00443, .001464); T[7][8]['C'] = 1.05
    T.add_edge(8,9); T[8][9]['z'] = (.006413, .004608); T[8][9]['C'] = 1.05
    T.add_edge(9,10); T[9][10]['z'] = (.006501, .004608); T[9][10]['C'] = 1.05
    T.add_edge(10,11); T[10][11]['z'] = (.001224, .000405); T[10][11]['C'] = 1.05
    T.add_edge(11,12); T[11][12]['z'] = (.002331, .000771); T[11][12]['C'] = 1.05
    T.add_edge(12,13); T[12][13]['z'] = (.009141, .007192); T[12][13]['C'] = 0.5
    T.add_edge(13,14); T[13][14]['z'] = (.003372, .004439); T[13][14]['C'] = 0.45
    T.add_edge(14,15); T[14][15]['z'] = (.00368, .003275); T[14][15]['C'] = 0.3
    T.add_edge(15,16); T[15][16]['z'] = (.004647, .003394); T[15][16]['C'] = 0.25
    T.add_edge(16,17); T[16][17]['z'] = (.008026, .010716); T[16][17]['C'] = 0.25
    T.add_edge(17,18); T[17][18]['z'] = (.004558, .003574); T[17][18]['C'] = 0.1
    T.add_edge(2,19); T[2][19]['z'] = (.001021, .000974); T[2][19]['C'] = 0.5
    T.add_edge(19,20); T[19][20]['z'] = (.009366, .00844); T[19][20]['C'] = 0.5
    T.add_edge(20,21); T[20][21]['z'] = (.00255, .002979); T[20][21]['C'] = 0.21
    T.add_edge(21,22); T[21][22]['z'] = (.004414, .005836); T[21][22]['C'] = 0.11
    T.add_edge(3,23); T[3][23]['z'] = (.002809, .00192); T[3][23]['C'] = 1.05
    T.add_edge(23,24); T[23][24]['z'] = (.005592, .004415); T[23][24]['C'] = 1.05
    T.add_edge(24,25); T[24][25]['z'] = (.005579, .004366); T[24][25]['C'] = 0.5
    T.add_edge(6,26); T[6][26]['z'] = (.001264, .000644); T[6][26]['C'] = 1.5
    T.add_edge(26,27); T[26][27]['z'] = (.00177, .000901); T[26][27]['C'] = 1.5
    T.add_edge(27,28); T[27][28]['z'] = (.006594, .005814); T[27][28]['C'] = 1.5
    T.add_edge(28,29); T[28][29]['z'] = (.005007, .004362); T[28][29]['C'] = 1.5
    T.add_edge(29,30); T[29][30]['z'] = (.00316, .00161); T[29][30]['C'] = 1.5
    T.add_edge(30,31); T[30][31]['z'] = (.006067, .005996); T[30][31]['C'] = 0.5
    T.add_edge(31,32); T[31][32]['z'] = (.001933, .002253); T[31][32]['C'] = 0.5
    T.add_edge(32,33); T[32][33]['z'] = (.002123, .003301); T[32][33]['C'] = 0.1
    T.add_edge(8,34); T[8][34]['z'] = (.012453, .012453); T[8][34]['C'] = 0.5
    T.add_edge(9,35); T[9][35]['z'] = (.012453, .012453); T[9][35]['C'] = 0.5
    T.add_edge(12,36); T[12][36]['z'] = (.012453, .012453); T[12][36]['C'] = 0.5
    T.add_edge(18,37); T[18][37]['z'] = (.003113, .003113); T[18][37]['C'] = 0.5
    T.add_edge(25,38); T[25][38]['z'] = (.003113, .003113); T[25][38]['C'] = 0.1

    nx.set_edge_attributes(T, 'l', {k: 0 for k in T.edges()})  # current magnitude square
    nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    nx.set_edge_attributes(T, 'K', {k: [] for k in T.edges()})  # customers whos demands pass through edge e
    nx.set_node_attributes(T, 'v', {k: 0 for k in T.nodes()})  # voltage magnitude square
    nx.set_edge_attributes(T, 'L', {k: 0 for k in T.edges()})  # loss upper bound
    nx.set_node_attributes(T, 'depth', {k: 0 for k in T.nodes()})

    leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    max_depth = 0
    for i in T.nodes()[1:]:
        T.node[i]['depth'] = len(nx.shortest_path(T, 0, i)) - 1
        if T.node[i]['depth'] > max_depth: max_depth = T.node[i]['depth']
    print 'max depth  = ', max_depth
    
    
    ####### method 1: pessimistic
    if method == "method1":
        max_loss_per = 0
        for e in T.edges():
            print "=== edge %s ==="%str(e)
            C = T[e[0]][e[1]]['C']  
            z = T[e[0]][e[1]]['z']  
            print 'C   = %.4f , |z| = %.6f'%(C,np.sqrt(z[0]**2 + z[1]**2) )
            if e[0] == 0: # first edge 
                #T[e[0]][e[1]]['L'] = np.sqrt(z[0]**2 + z[1]**2)*  C**2
                T[e[0]][e[1]]['L'] = (max_depth - T.node[e[1]]['depth']) * np.sqrt(z[0]**2 + z[1]**2)*  C**2
            else:
                T[e[0]][e[1]]['L'] = (max_depth - T.node[e[1]]['depth']) * np.sqrt(z[0]**2 + z[1]**2)*  C**2
                #T[e[0]][e[1]]['L'] =  np.sqrt(z[0]**2 + z[1]**2)*  C**2
            per = T[e[0]][e[1]]['L']/C * 100
            if per > max_loss_per: max_loss_per = per
            print 'loss  = %.6f\t %4.2f%%'%( T[e[0]][e[1]]['L'], per)
        #print 'max %% = %.2f'%max_loss_per
    elif method == "method2":
        ####### method 2: ######
        print "##### method 2 #####"
        for e in T.edges():
            T_ = T.copy()
            if e[0] != 0: #not root, then delete the dege connected to e[0]
                # disconnect the graph at e[0]
                parent = nx.predecessor(T,0,e[0])[0]
                T_.remove_node(parent)
                T_ = nx.bfs_tree(T_,e[0])
            C = T[e[0]][e[1]]['C']
            z_max = {d:[] for d in range(1,max_depth - T.node[e[0]]['depth'] + 1)}
            for i in T_.nodes():
                if i != e[0]:
                    parent = nx.predecessor(T_,e[0],i)[0]
                    # print "(%d,%d)"%(parent,i), " resistance is ", np.sqrt(T[parent][i]['z'][0]**2 + T[parent][i]['z'][1]**2)
                    depth = T.node[i]['depth'] - T.node[e[0]]['depth']
                    z_max[depth].append(np.sqrt(T[parent][i]['z'][0]**2 + T[parent][i]['z'][0]**2))
            z_max = {l:max(v) for l,v in z_max.iteritems() if v != []}
            loss1 = sum([C**2 * z for l,z in z_max.iteritems()]) # very bad
            loss2 = 0
            for d in z_max.keys():
                loss2 += z_max[d] * (C-loss2)**2
            #print "loss of %-8s  = %.6f"%(str(e), loss1)
            T[e[0]][e[1]]['L'] = loss2
            print "loss of %-8s  = %.6f\t %4.2f%%"%(str(e), loss2, loss2/C)
    elif method == "fixed":
        for e in T.edges():
            C = T[e[0]][e[1]]['C']  
            z = T[e[0]][e[1]]['z']  
            T[e[0]][e[1]]['L'] = C*loss_ratio
            T[e[0]][e[1]]['C_'] = C*(1-loss_ratio)
            print "=== edge %s ==="%str(e)
            print 'C     = %.4f , |z| = %.6f'%(C,np.sqrt(z[0]**2 + z[1]**2) )
            print 'loss  = %.6f\t %4.2f%%'%( T[e[0]][e[1]]['L'], T[e[0]][e[1]]['L']/C * 100)

#    for e in T.edges():
#        children = T[e[1]].keys()
#        if children[0] == e[0]: children = children[1:]
#        print 'edge %s children'%str(e), children







    if file != '':
        nx.write_gml(T, file_name)
        nx.write_gpickle(T, file_name)


    if visualize:
        pos=nx.graphviz_layout(T,prog='dot')
        nx.draw(T, pos)
        node_labels = {k:k for k in T.nodes()}
        nx.draw_networkx_labels(T, pos, labels = node_labels)
        edge_labels = nx.get_edge_attributes(T,'z')
        nx.draw_networkx_edge_labels(T, pos, labels = edge_labels)
        plt.show()

    return T

def rnd_instance_from_graph(T_file_name='data/38_node_test.gpickle', n=3, v_0=1, v_max=1.21, v_min=0.81000,
                      load_theta_range=(-0.628, 0.628), max_load = .1,
                      util_func=lambda x, y: x - x + np.random.rand()):
    T = nx.read_gpickle(T_file_name)

    # initialize
    for k in T.nodes():
        T.node[k]['N'] = []  # customers on node i
        T.node[k]['v'] = 0   # voltage

    ins = maxOPF_instance()
    T.node[0]['v'] = ins.v_0
    ins.topology = T
    ins.leaf_nodes = T.graph['leaf_nodes']


    ins.leaf_edges = T.graph['leaf_edges']

    ins.I = np.arange(n)
    ins.v_0 = v_0
    ins.v_max = v_max
    ins.v_min = v_min
    ins.V_ = (v_0 - v_min) / 2
    set_customers(ins, n=n, load_theta_range=load_theta_range, max_load=max_load, util_func=util_func)


    return ins


# topology must be filled in instance ins
def set_customers(ins, n=3, load_theta_range=(0,0.628), max_load=2,
                  util_func=lambda x, y: x - x + np.random.rand()):
    T = ins.topology
    ins.leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    ins.leaf_edges = [(i, nx.predecessor(T, i, 0)[0]) for i in ins.leaf_nodes]
    ins.n = n
    ins.loads_S = np.random.rand(n) * max_load
    ins.loads_angles = np.random.uniform(load_theta_range[0], load_theta_range[1], n) * load_theta_range[0]
    ins.loads_utilities = np.array(map(util_func, ins.loads_S, ins.loads_angles))
    ins.loads_P = np.array(map(lambda x, t: x * np.math.cos(t), ins.loads_S, ins.loads_angles))
    ins.loads_Q = np.array(map(lambda x, t: x * np.math.sin(t), ins.loads_S, ins.loads_angles))
    ins.max_load_theta = max(ins.loads_angles)

    for k in range(n):
        attached_node = np.random.choice(T.nodes()[1:])
        T.node[attached_node]['N'].append(k)
        ins.customer_node[k] = attached_node
        ins.customer_path_nodes[k] = nx.shortest_path(T, 0, attached_node)
        ins.customer_path[k] = zip(ins.customer_path_nodes[k], ins.customer_path_nodes[k][1:])
        for e in ins.customer_path[k]: T[e[0]][e[1]]['K'].append(k)

        #########
        logging.debug('------- customer: %d -------' % k)
        logging.debug('demand: (%f,%f)' % (ins.loads_P[k], ins.loads_Q[k]))
        logging.debug('attached bus: %d' % attached_node)
        logging.debug('path: %s' % str(ins.customer_path_nodes[k]))
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

            logging.debug("shared path with leaf %d: %s" % (l, str(ins.customer_leaf_path_nodes[(k, l)])))
            logging.debug("shared path with leaf %d: %s" % (l, str(ins.customer_leaf_path[(k, l)])))
            ins.Q[(k, l)] = 0
            tmp_str = "path impedance through leaf %d: " % l
            for e in ins.customer_leaf_path[(k, l)]:
                ins.Q[(k, l)] += T[e[0]][e[1]]['z'][0] * ins.loads_P[k] + T[e[0]][e[1]]['z'][1] * ins.loads_Q[k]
                tmp_str += '(%f, %f) ' % (T[e[0]][e[1]]['z'][0], T[e[0]][e[1]]['z'][1])
            logging.debug(tmp_str)
            logging.debug('Q[(%d,%d)]=%f' % (k, l, ins.Q[(k, l)]))


def print_instance(ins):
    print "===== I N S T A N C E  D I S C ====="
    print "-- netork: "
    print ' # nodes: ', len(ins.topology.nodes())
    print ' # edges: ', len(ins.topology.edges())
    # print ' Edges:', ins.topology.edges()
    print ' Leaf nodes:', ins.leaf_nodes
    print "-- customers:"
    print(' util:       %s' % str({k: round(ins.loads_utilities[k],2) for k in range(ins.n)}) ) #str(map(lambda x: round(x, 2), ins.loads_utilities)))
    print(' loads_S:    %s' % str({k: round(ins.loads_S[k],3) for k in range(ins.n)}) ) #str(map(lambda x: round(x, 2), ins.loads_S)))
    print(' loads_P:    %s' % str({k: round(ins.loads_P[k],3) for k in range(ins.n)}) ) #str(map(lambda x: round(x, 2), ins.loads_S)))
    print(' loads_Q:    %s' % str({k: round(ins.loads_Q[k],3) for k in range(ins.n)}) ) #str(map(lambda x: round(x, 2), ins.loads_S)))
    print(' Q(k,l):     %s' % str({i[0]: round(i[1], 3) for i in ins.Q.iteritems()}))
    print "-- cons: "
    print(' C: %s' % {e: round(ins.topology[e[0]][e[1]]['C'], 3) for e in ins.topology.edges()})
    print(' z: %s' % {e: (round(ins.topology[e[0]][e[1]]['z'][0], 3), round(ins.topology[e[0]][e[1]]['z'][1], 3)) for e in ins.topology.edges()})
    print ' V_ = ', (ins.v_0 - ins.v_min) / 2
    print "===================================="


def print_customer(ins, k):
    print('------- customer: %d -------' % k)
    print('demand: (%f,%f)' % (ins.loads_P[k], ins.loads_Q[k]))
    print 'Q values: ', {l: ins.Q[k:l] for l in ins.leaf_nodes}
    print('attached bus: %d' % ins.customer_node[k])
    print('path: %s' % str(ins.customer_path_nodes[k]))


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

    network_38node()
    # ins = rnd_instance_from_graph()
    # print_instance(ins)
