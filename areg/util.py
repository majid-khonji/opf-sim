__author__ = 'akarapetyan'
import math
import random
import networkx as nx


def deletewithProbability():

    r = random.random()
    if r <= 0.5:
        return True
    else:
        return False


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

    nx.set_edge_attributes(TheMicrogrid, 'Capacity', 0)
    nx.set_edge_attributes(TheMicrogrid, 'label', 'None')
    nx.set_node_attributes(TheMicrogrid, 'color', 'blue')
    nx.set_node_attributes(TheMicrogrid, 'label', 'DG')

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


def constrained_sum_sample_pos(n, total):
    #Return a randomly chosen list of n positive integers summing to total.
    #Each such list is equally likely to occur.

    dividers = sorted(random.sample(xrange(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def generate_Users(Total, number, edge, Power, Correlation):

    TempList = constrained_sum_sample_pos(number, Total)
    TempUsers = []
    for i in TempList:
        if Power == "F":
            DemandImaginary = round(i * math.tan(random.randrange(0, 36) * math.pi/180), 3)
        else:
            DemandImaginary = 0

        TotalDemand = math.sqrt(math.pow(DemandImaginary, 2) + math.pow(i, 2))
        if Correlation == "C":
            Utility = round(math.pow(TotalDemand, 2)/math.pow(10, 5))
        elif Correlation == "U":
            Utility = random.randrange(0, 1000)+1
        elif Correlation == "L":
            if edge == 1:
                Utility = TotalDemand/math.pow(10, 3)
            else:
                Utility = TotalDemand/math.pow(10, 3) + 30

        TempUsers.append([Utility, i, DemandImaginary, edge])
    return TempUsers

def CustomersGenerator(TheMicrogrid, Power, Correlation, CustomerType, tokos):

    Customers = []

    #Distributing Customers
    for node in TheMicrogrid.nodes_iter():
            if node == 0:

                for x in TheMicrogrid.edges([node]):
                    origin, destination = x

                    TheDemand = 2500000
                    if CustomerType == "M":
                        number = 6
                    else:
                        number = int(TheDemand/4000)

                    TheMicrogrid.node[destination]['label'] = (" Bus %d - %.2f W" % (destination, TheDemand))
                    Customers.extend(generate_Users(int(TheDemand), number, destination, Power, Correlation))

            else:
                for x in TheMicrogrid.edges([node]):
                    origin, destination = x
                    if destination > origin:
                        TheDemand = TheMicrogrid[origin][destination]['label'].split(":")
                        TheDemand = TheDemand[1][:-1]
                        TheDemand = float(TheDemand)
                        AddLoads = tokos/100.
                        TheDemand = TheDemand + (TheDemand*AddLoads)
                        TheDemand = int(TheDemand/4)
                        TempStore = []
                        number = int(round(TheDemand/4000))
                        for i in nx.all_simple_paths(TheMicrogrid, 0, destination):
                            TempStore = i
                        TempStore.remove(0)
                        TheMicrogrid.node[destination]['label'] = (" Bus %d - %.2f W" % (destination, TheDemand))
                        Customers.extend(generate_Users(int(TheDemand), number, TempStore,  Power, Correlation))

    return Customers



