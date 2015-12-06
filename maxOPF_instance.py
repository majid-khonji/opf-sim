__author__ = 'majid'
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import logging


class maxOPF_instance(object):
    topology = None
    leaf_edges = {} # leaf edges
    v_0 = 0
    v_max = 0
    v_min = 0
    n = 0 # number of customers
    Q = {}  # Q value of each customer. Dictionary keyed by (k,i) where k is customer and i is leaf vertex
    customer_path = {}  # path of customer k
    customer_leaf_path = {}  # path of customer k
    customer_node = {} # customer to node
    max_load_theta = 0
    loads_S = []
    loads_P = []
    loads_Q = []
    loads_angles = []
    loads_utilities = []

    customer_path_nodes = {}  # path of customer k
    customer_leaf_path_nodes = {}  # path of customer k
    leaf_nodes = {}  # leaf nodes

class maxOPF_sol(object):
    def __init__(self):
        self.obj = 0; self.idx = []; self.running_time = 0; ar = None  # aproximation ratio
        self.topology = None # updated graph with voltage, current,..etc filled

def rnd_instance(n=3, node_count=3, v_0=1, v_max=1.05, v_min=.95, capacity_range=(100, 200), max_load_theta=0.628,
                 max_imp_theta=0.628,
                 max_load=2,
                 util_func=lambda x, y: x - x + np.random.rand()):
    ins = maxOPF_instance()
    ins.v_0 = v_0
    ins.v_max = v_max
    ins.v_min = v_min
    rnd_topology(ins, node_count=node_count, capacity_range=capacity_range, max_imp_theta=max_imp_theta)
    set_customers(ins, n=n, max_load_theta=max_load_theta, max_load=max_load, util_func=util_func)
    return ins


# inserts ins.v_0 into the root of the topology ins.topology
def rnd_topology(ins, node_count=3, capacity_range=(100, 200), max_imp_theta=0.628):
    # T = nx.path_graph(node_count)
    T = nx.DiGraph()
    T.add_path(range(node_count))
    nx.set_edge_attributes(T, 'C', {k: 100 for k in T.edges()})  # max VA capacity
    nx.set_edge_attributes(T, 'l', {k: 0 for k in T.edges()})  # current magnitude square
    nx.set_edge_attributes(T, 'S', {k: (0, 0) for k in T.edges()})  # edge power [Complex num]
    nx.set_edge_attributes(T, 'z', {k: (.1, .1) for k in T.edges()})  # impedence [complex
    nx.set_node_attributes(T, 'v', {k: 0 for k in T.nodes()})  # voltage magnitude square
    T.node[0]['v'] = ins.v_0
    ins.leaf_nodes = [l for l, d in T.degree().items() if d == 1][1:]
    ins.leaf_edges = [(i, T.predecessors(i)[0]) for i in ins.leaf_nodes]

    ins.topology = T
    # nx.set_node_attributes(T, 's', {k: 'DG' for k in T.nodes()})  # load
    # pos = nx.spring_layout(T)
    # nx.draw(T, pos)
    # node_labels = nx.get_node_attributes(T,'customers')
    # nx.draw_networkx_labels(T, pos, labels = node_labels)
    # edge_labels = nx.get_edge_attributes(T,'z')
    # nx.draw_networkx_edge_labels(T, pos, labels = edge_labels)
    # print pos
    # plt.savefig('this.png')
    # plt.show()


# topology must be filled in instance ins
def set_customers(ins, n=3, max_load_theta=0.628, max_load=2,
                  util_func=lambda x, y: x - x + np.random.rand()):
    T = ins.topology
    ins.n = n
    ins.loads_S = np.random.rand(n) * max_load
    ins.loads_angles = np.random.rand(n) * max_load_theta
    ins.loads_utilities = np.array(map(util_func, ins.loads_S, ins.loads_angles))
    ins.loads_P = np.array(map(lambda x, t: x * np.math.cos(t), ins.loads_S, ins.loads_angles))
    ins.loads_Q = np.array(map(lambda x, t: x * np.math.sin(t), ins.loads_S, ins.loads_angles))
    ins.max_load_theta = max(ins.loads_angles)

    node_customers = {l:[] for l in T.nodes()}
    for k in range(n):
        attached_node = np.random.choice(T.nodes()[1:])
        node_customers[attached_node].append(k)
        T.node[attached_node]['customers'] = node_customers[attached_node]
        ins.customer_node[k] = attached_node
        ins.customer_path_nodes[k] = nx.shortest_path(T, 0, attached_node)
        ins.customer_path[k] = zip(ins.customer_path_nodes[k], ins.customer_path_nodes[k][1:])
        #########
        logging.debug('------- customer: %d -------' % k)
        logging.debug('demand: (%f,%f)' % (ins.loads_P[k], ins.loads_Q[k]))
        logging.debug('attached bus: %d' % attached_node)
        logging.debug('path: %s' % str(ins.customer_path[k]))
        #########
        for l in ins.leaf_nodes:
            ins.customer_leaf_path[(k, l)] = []
            leaf_path = nx.shortest_path(T, 0, l)
            ins.customer_leaf_path_nodes[(k, l)] = [i for i in range(np.min([len(leaf_path), len(ins.customer_path_nodes[k])]))
                                              if leaf_path[i] == ins.customer_path_nodes[k][i]]
            ins.customer_leaf_path[(k,l)] = zip(ins.customer_leaf_path_nodes[(k,l)], ins.customer_leaf_path_nodes[(k,l)][1:])

            logging.debug("shared path with leaf %d: %s" % (l, str(ins.customer_leaf_path_nodes[(k, l)])))
            ins.Q[(k, l)] = 0
            tmp_str = "path impedance through leaf %d: " % l
            for i in ins.customer_leaf_path_nodes[(k, l)][1:]:
                parent = T.predecessors(i)[0]
                ins.Q[(k, l)] += T[parent][i]['z'][0] * ins.loads_P[k] + T[parent][i]['z'][1] * ins.loads_Q[k]
                tmp_str += '(%f, %f) ' % (T[parent][i]['z'][0], T[parent][i]['z'][1])
            logging.debug(tmp_str)
            logging.debug('Q[(%d,%d)]=%f' % (k, l, ins.Q[(k, l)]))


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    rnd_instance()
    # plt.show()
