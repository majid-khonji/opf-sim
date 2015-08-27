__author__ = 'akarapetyan'
import simpy
import util
import time
import math
import networkx as nx
import gurobipy as gbp
import copy


class MicroGrid():
    def __init__(self, env, Capacity, Data, EdgeCapacities, TheMicrogrid, Epsilon, SimulationTime, CustomerType, Power,
                 Correlation):
        self.env = env
        self.CustomerType = CustomerType
        self.TheMicrogrid = TheMicrogrid
        self.Power = Power
        self.Correlation = Correlation
        self.Capacity = Capacity
        self.Data = Data
        self.Epsilon = Epsilon
        self.Capacities = EdgeCapacities
        self.GreedyTreeData = []
        self.GreedyTreeData = copy.deepcopy(Data)

        self.tokos = [1.0] * 19
        self.tokos.extend([3] * 20)
        self.tokos.extend([5] * 20)
        self.tokos.extend([7] * 20)
        self.tokos.extend([9] * 20)
        self.tokos.extend([11] * 20)

        # Microgrid Initialization
        f.write('\n')
        f.write('The Microgrid started to operate with the following settings;')
        f.write('\n')
        f.write('Starting Time: %d' % env.now)
        f.write('\n')
        f.write('Capacity: %d' % self.Capacity)
        f.write('\n')
        f.write('Connected Devices: %d' % len(self.Data))
        f.write('\n')
        f.write('Scheduled Maximization Interval: 5 minutes')
        f.write('\n')
        f.write('Overall Utility of the Connected Devices: %f' % self.getOverallUtility())
        f.write('\n')
        f.write('Overall Demand of the Connected Devices: %f' % self.getOverallDemand())
        f.write('\n')
        f.write('Simulation Duration: %d' % SimulationTime)
        f.write('\n')
        f.write('User\'s Demands and Utilities Correlation: Correlated')
        f.write('\n')
        f.write('----------------------------------------------------------------------')
        f.write('\n')
        # Start the run process every time an instance is created.
        self.action = env.process(self.run())

    def getOverallUtility(self):
        OverallUtility = 0
        for i in self.Data:
            OverallUtility += i[0]
        return OverallUtility

    def getOverallDemand(self):
        OverallDemand = 0
        OverallDemand += math.sqrt(
            math.pow(sum([dem[1] for dem in self.Data]), 2) + math.pow(sum([dem[2] for dem in self.Data]), 2))
        return OverallDemand

    def run(self):

        while True:
            f.write('Scheduled maximization procedure started with the following settings;')
            f.write('\n')
            f.write('Starting Time: %d' % env.now)
            f.write('\n')
            f.write('Capacity: %d' % self.Capacity)
            f.write('\n')
            f.write('Connected Devices: %d' % len(self.Data))
            f.write('\n')
            f.write('Scheduled Maximization Interval: 5 minutes')
            f.write('\n')
            f.write('Overall Utility of the Connected Devices: %f' % self.getOverallUtility())
            f.write('\n')
            f.write('Overall Demand of the Connected Devices: %f' % self.getOverallDemand())
            f.write('\n')
            # Let's start the Maximization procedures
            yield self.env.process(self.Maximize())
            # Maximization Interval

            # Number of users
            if env.now > 0:
                theindex = (env.now / 5) - 1
            else:
                theindex = 0

            self.Data = []
            self.Data = util.CustomersGenerator(self.TheMicrogrid, self.Power, self.Correlation, self.CustomerType,
                                                self.tokos[theindex])
            self.GreedyTreeData = []
            self.GreedyTreeData = copy.deepcopy(self.Data)
            f.write('----------------------------------------------------------------------')
            f.write('\n')

    def Maximize(self):
        whethertogo = self.GurobiOPT()
        if whethertogo:
            self.GreedyTree()
            TimeOut = 5
            yield self.env.timeout(TimeOut)
        else:
            yield self.env.timeout(0)

    def findDemands(self, edge, Data):
        edgeDemand = []
        for idx, customer in enumerate(Data):
            if isinstance(customer[3], list):
                if edge in customer[3]:
                    edgeDemand.append(customer)
            elif edge == customer[3]:
                edgeDemand.append(customer)
        return edgeDemand

    def GurobiOPT(self):
        StartingTime = time.time()
        solutionStatus = "No Solution"
        try:
            # Create a new model
            m = gbp.Model("miqcp")
            x = [None] * len(self.Data)
            # Create variables
            for i in range(len(self.Data)):
                x[i] = m.addVar(vtype=gbp.GRB.BINARY, name="x[%d]" % i)

            # Integrate new variables
            m.update()

            # Set objective: x
            obj = gbp.quicksum([x[c] * self.Data[c][0] for c in range(len(self.Data))])
            m.setObjective(obj, gbp.GRB.MAXIMIZE)

            count = 0
            for edge in self.TheMicrogrid.nodes_iter():
                curcustomers = []
                for idx, customer in enumerate(self.Data):
                    if isinstance(customer[3], list):
                        if edge in customer[3]:
                            curcustomers.append((idx, customer))
                    else:
                        if edge == customer[3]:
                            curcustomers.append((idx, customer))

                if len(curcustomers) > 0:
                    constraint = gbp.quicksum([x[c[0]] * c[1][1] for c in curcustomers]) * gbp.quicksum(
                        [x[c[0]] * c[1][1] for c in curcustomers]) + gbp.quicksum(
                        [x[c[0]] * c[1][2] for c in curcustomers]) * gbp.quicksum(
                        [x[c[0]] * c[1][2] for c in curcustomers])
                    m.addConstr(constraint <= self.Capacities[edge] * self.Capacities[edge], "qc%d" % count)
                    count += 1
                    m.update()

            m.setParam("TimeLimit", 100)
            m.setParam("MIPGapAbs", 0.000001)
            # m.setParam("SolutionLimit", 1)
            m.setParam("IntFeasTol", 0.000000001)
            m.setParam("OutputFlag", 0)

            m.update()
            m.optimize()

            if m.status == gbp.GRB.status.OPTIMAL:
                # MaximizedUtility = obj.getValue()
                solutionStatus = m.status
            else:
                solutionStatus = "No Solution"

            MaximizedUtility = obj.getValue()
            solutionStatus = m.status

            # for v in m.getVars():
            # print('%s %g' % (v.varName, v.x))
        except gbp.GurobiError as e:
            pass

        if solutionStatus != "No Solution":
            TimeElapsed = time.time() - StartingTime
            f.write('\n')
            f.write('Gurobi\'s Maximization Algorithm\'s Results;')
            f.write('\n')
            f.write('Maximized Utility: %f' % MaximizedUtility)
            f.write('\n')
            f.write('Running time: %f seconds' % TimeElapsed)
            f.write('\n')
            j = open("gurobi.txt", 'a')
            j.write("%f,%f,%d,%f,%s \n" % (
                MaximizedUtility, TimeElapsed, len(self.Data), self.getOverallUtility(), solutionStatus))
            j.close()
            return True
        else:
            return False

    def GreedyTree(self):

        StartingTime = time.time()
        tempdata = copy.deepcopy(self.GreedyTreeData)
        tempdata = list(enumerate(tempdata))

        tempdata = sorted(tempdata, key=lambda tup: tup[1][0], reverse=True)
        L = tempdata[0][1][0] / math.pow(len(self.GreedyTreeData), 2)

        for i in range(len(tempdata)):
            tempdata[i][1][0] = math.floor(tempdata[i][1][0] / L)

        libertador = int(2 * math.log(len(self.GreedyTreeData), 2)) + 1
        N = [None] * libertador
        N[0] = []

        for k in tempdata:
            if k[1][0] >= 0 and k[1][0] < 2:
                N[0].append(self.GreedyTreeData[k[0]])

        for i in range(1, libertador):
            N[i] = []
            for k in tempdata:
                if k[1][0] >= math.pow(2, i) and k[1][0] < math.pow(2, i + 1):
                    N[i].append(self.GreedyTreeData[k[0]])

        tempans = []
        for i in N:
            if len(i) > 0:
                tempans.append(self.GreedyMagnitude(i))

        TimeElapsed = time.time() - StartingTime
        MaximisedUtility = max(tempans)

        f.write('\n')
        f.write('Greedy Logarithmic Maximization Algorithm\'s Results;')
        f.write('\n')
        f.write('Maximized Utility: %f' % MaximisedUtility)
        f.write('\n')
        f.write('Running time: %f seconds' % TimeElapsed)
        f.write('\n')
        j = open("glplotdata.txt", 'a')
        j.write("%f,%f,%d,%f \n" % (MaximisedUtility, TimeElapsed, len(self.Data), self.getOverallUtility()))
        j.close()

    def GreedyMagnitude(self, customerset):

        SortedData = sorted(customerset, key=lambda tup: math.sqrt(math.pow(tup[1], 2) + math.pow(tup[2], 2)))
        MaximizedUtility = 0
        Set = []
        for idx, customer in enumerate(SortedData):
            flag = True
            for idx2, i in enumerate(Set):
                tempcalc = 0
                if isinstance(i[3], list):
                    trigger = True
                    for edge in i[3]:
                        tempDemands = self.findDemands(edge, Set)
                        tempDemands.append(customer)
                        tempcalc += math.pow(sum([dem[1] for dem in tempDemands]), 2) + math.pow(
                            sum([dem[2] for dem in tempDemands]), 2)
                        if tempcalc > math.pow(self.Capacities[edge], 2):
                            trigger = False
                            break

                    if not trigger:
                        flag = False
                        break
                else:
                    tempDemands = self.findDemands(i[3], Set)
                    tempDemands.append(customer)
                    tempcalc += math.pow(sum([dem[1] for dem in tempDemands]), 2) + math.pow(
                        sum([dem[2] for dem in tempDemands]), 2)
                    if tempcalc > math.pow(self.Capacities[i[3]], 2):
                        flag = False
                        break

            if flag:
                Set.append(customer)
                MaximizedUtility += customer[0]

        return MaximizedUtility


# PARAMETERS
TotalCapacity = 5000000
EPSILON = 0.1
SimulationTime = 600
env = simpy.Environment()

Power = "A"
Correlation = "C"
CustomerType = "M"

TheMicrogrid, EdgeCapacities = util.graphGenerator(TotalCapacity)
Customers = util.CustomersGenerator(TheMicrogrid, Power, Correlation, CustomerType, 1.0)


# Plot the Graph
A = nx.to_agraph(TheMicrogrid)
A.layout('dot', args='-Nfontsize=10 -Nwidth=".4" -Nheight=".4" -Nmargin=0 -Gfontsize=8 -Earrowhead=none -Gnodesep=1 ')
A.draw('test.png')



# Output Buffer
Filename = "Results-" + str(int(time.time())) + ".txt"
f = open(Filename, 'w+')

Microgrid = MicroGrid(env, TotalCapacity, Customers, EdgeCapacities, TheMicrogrid, EPSILON, SimulationTime,
                      CustomerType, Power, Correlation)
env.run(until=SimulationTime)
f.close()
