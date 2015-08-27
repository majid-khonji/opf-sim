__author__ = 'majid'
import networkx as nx
import random
import numpy as np
from ckp import ckp_instance


def tree_gen(max_children=3, max_depth = 3):
    T = nx.balanced_tree(max_children,max_depth)


    nx.set_edge_attributes(T, 'capacity', {k:0 for k in T.edges()})
    nx.set_edge_attributes(T, 'l', {k:(0,0) for k in T.edges()}) # current mangnitude square
    nx.set_node_attributes(T, 'S', {k:(0,0) for k in T.nodes()}) #edge power
    nx.set_node_attributes(T, 'z', {k:(0,0) for k in T.nodes()}) #impedence
    nx.set_node_attributes(T, 's', {k:'DG' for k in T.nodes()}) #load
    return T

def graphGenerator(TotalCapacity):

    TheMicrogrid = nx.Graph()
    TheMicrogrid.add_nodes_from([0, 1])
    TheMicrogrid.add_edge(0, 1)

    secondlvlnodes = random.randint(2, 4)
    currentsize = len(TheMicrogrid)
    for i in range(currentsize, currentsize + secondlvlnodes):
        TheMicrogrid.add_node(i)
        TheMicrogrid.add_edge(1, i)

    for i in range(currentsize, currentsize + secondlvlnodes):
        tempqty = len(TheMicrogrid)
        thirdlvlnodes = random.randint(2, 3)
        for j in range(tempqty, tempqty + thirdlvlnodes):
            TheMicrogrid.add_node(j)
            TheMicrogrid.add_edge(i, j)

    TempMicrogrid = TheMicrogrid.copy()
    for nodes in TempMicrogrid.nodes_iter():
        if TheMicrogrid.degree(nodes) == 1 and nodes != 0:
            tempqty = len(TheMicrogrid)
            thirdlvlnodes = random.choice([0, 0, 2])
            for j in range(tempqty, tempqty + thirdlvlnodes):
                TheMicrogrid.add_node(j)
                TheMicrogrid.add_edge(nodes, j)

    nx.set_edge_attributes(TheMicrogrid, 'Capacity', {k:0 for k in TheMicrogrid.edges()})
    nx.set_edge_attributes(TheMicrogrid, 'label', {k:'none' for k in TheMicrogrid.edges()})
    nx.set_node_attributes(TheMicrogrid, 'color', {k:'blue' for k in TheMicrogrid.nodes()})
    nx.set_node_attributes(TheMicrogrid, 'label', {k:'DG' for k in TheMicrogrid.nodes()})


    EdgeCapacities = {}
    Leaves = []

    #Distributing Capacities
    for node in TheMicrogrid.nodes_iter():
        toContinue = True
        TempArray = []
        if node == 0:
            TempArray = [TotalCapacity]
            TheMicrogrid.node[node]['color'] = 'red'
        elif len(TheMicrogrid.edges([node])) > 1:
            for z in TheMicrogrid.edges([node]):
                current, parent = z
                if parent < current:
                    break
            TempArray = [TheMicrogrid[parent][node]['Capacity']/(len(TheMicrogrid.edges([node]))-1)]*(len(TheMicrogrid.edges([node]))-1)
            TheMicrogrid.node[node]['color'] = 'blue'
        elif TheMicrogrid.degree(node) == 1:
            Leaves.append(node)
            toContinue = False

        if toContinue:
            counter = 0

            if node == 0:
                for x in TheMicrogrid.edges([node]):
                    origin, destination = x
                    TheMicrogrid[origin][destination]['Capacity'] = TempArray[counter]
                    TheMicrogrid[origin][destination]['label'] = ("Edge %d: %d W" % (destination, TempArray[counter]))
                    EdgeCapacities[destination] = TempArray[counter]
                    counter += 1

            else:
                for x in TheMicrogrid.edges([node]):
                    origin, destination = x
                    if destination > origin:
                        TheMicrogrid[origin][destination]['Capacity'] = TempArray[counter]
                        TheMicrogrid[origin][destination]['label'] = ("Edge %d: %d W" % (destination, TempArray[counter]))
                        EdgeCapacities[destination] = TempArray[counter]
                        counter += 1
    return TheMicrogrid, EdgeCapacities





if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # T = tree_gen(2,2)
    # nx.draw_networkx(T)
    # plt.show()
    ins = ckp_instance_rnd()
    print ins.loads_utilities
