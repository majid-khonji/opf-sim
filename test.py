#!/usr/bin/python
__author__ = 'mkhonji'
import instance as i
import s_maxOPF_algs as s
import OPF_algs as o
import numpy as np
import util as u
import logging
import sys
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
# root = logging.getLogger()
# root.setLevel(logging.INFO)
# ch = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(levelname)s: %(message)s')
# ch.setFormatter(formatter)
# root.addHandler(ch)


sum_r1 = 0
sum_r2 = 0

sum_t1 = 0
sum_t2 = 0
sum_t3 = 0
t = 0
cons = ''
capacity_flag='C_'
for n in np.arange(100,1501,100):
    t +=1
    print '############# step %d ##########'% t
    T = i.network_38node(loss_ratio=.0)
    # ins = i.rnd_instance_from_graph(T, n=500)
    scenario = "FUR"
    F_percentage = .25
    ins = i.sim_instance(T, scenario=scenario,F_percentage=F_percentage, n=n, capacity_flag=capacity_flag)
    # print "scenario: %s  F%% %.2f"%(scenario, F_percentage)
    # ins.loads_utilities = np.ones(ins.n)
    # u.print_instance(ins)

    # print '----- OPT smp ------'
    # sol0 = s.OPT(ins, cons=cons, capacity_flag=capacity_flag)
    sol0 = s.adaptive_OPT(ins, cons=cons)
    print 'OPTsmp: obj value: ', sol0.obj
    # print 'time: %.4f sec' % sol0.running_time
    sum_t1 += sol0.running_time

    # print '-------- OPT -------'
    sol1 = o.max_OPF_OPT(ins, cons=cons)
    print 'OPT   : obj value: ', sol1.obj
    # print 'time: %.4f sec' % sol1.running_time
    sum_t2 += sol1.running_time

    # print '----- greedy ------'
    sol2 = None
    # if F_percentage == 0:
    #     sol2 = s.greedy(ins, cons=cons, capacity_flag='C_')
    # else:
    #     sol2 = s.mixed_greedy(ins, cons=cons, capacity_flag='C_')
    sol2 = s.adaptive_greedy(ins,cons=cons)
    # sol2 = s.greedy_card(ins,cons=cons)
    sol2.ar = sol2.obj / sol1.obj
    # print 'Greedy loss ratio = ', sol2.loss_ratio
    # if s.check_feasibility(ins,sol2.x)==False:
    #     print 'Greedy infeasible !!!'
    #     break
    print "GRD   : obj value:  %5.2f"%(sol2.obj)
    print 'again: %5.2f'% np.sum([ins.loads_utilities[k]*sol2.x[k] for k in range(ins.n)])
    print s.check_feasibility(ins,sol2.x,capacity_flag=capacity_flag)

    # print 'max sol idx = %s' % str(sol2.idx)
    # print '# groups: ', len(sol2.groups)
    # print 'time: %.4f sec' % sol2.running_time
    sum_t3 += sol2.running_time


    print '=== ratio %.3f (%.3f w.r. smp.) ===' % (sol2.ar, sol2.obj / sol0.obj)
    sum_r1 += sol2.ar;
    sum_r2 += sol2.obj / sol0.obj
    print ' avg r = %.3f (%.3f)' % (sum_r1 / t, sum_r2 / t)
    print ' avg t = g: %.3f | opt-s: %.3f | opt: %.3f' % (sum_t3 / t, sum_t1 / t, sum_t2 / t)

    # sol3 = o.min_loss_OPF(ins, sol0.x, cons=cons)
    # print 'OPT loss    = ', sol3.obj
    # sol4 = o.min_loss_OPF(ins, sol2.x, cons=cons)
    # print 'greedy loss = ', sol4.obj
    # print
    #
    # if sol4.obj == - np.inf or sol3.obj == - np.inf:
    #     print '!!! infeasible loss/voltage sol !!!'
    #     break
    # if sol0.obj < sol2.obj:
    #     print "!!! greedy > OPT(s) !!!"
    #     break
