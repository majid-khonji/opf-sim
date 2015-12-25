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




cons = ''

sum_r1 = 0
sum_r2 = 0

sum_t1 = 0
sum_t2 = 0
sum_t3 = 0
for t in range(1, 100000):
    T = i.network_38node(loss_ratio=.08)
    # ins = i.rnd_instance_from_graph(T, n=500)
    # ins.loads_utilities = ins.loads_utilities**2
    ins = i.sim_instance(T, scenario='FCI', n=1000)
    # u.print_instance(ins)

    print '----- OPT smp ------'
    # ins.F = ins.I
    # ins.I = []
    sol0 = s.OPT(ins, cons=cons, capacity_flag='C_')
    print 'obj value: ', sol0.obj
    # print 'obj idx: ', sol0.idx
    print 'time: %.4f sec' % sol0.running_time
    sum_t1 += sol0.running_time

    print '-------- OPT -------'
    sol1 = o.max_OPF(ins, cons=cons)
    print 'obj value: ', sol1.obj
    # print 'obj idx: ', sol1.idx
    print 'time: %.4f sec' % sol1.running_time
    sum_t2 += sol1.running_time

    print '----- greedy ------'
    sol2 = s.greedy(ins, cons=cons, capacity_flag='C_')
    sol2.ar = sol2.obj / sol1.obj
    print "max value =", sol2.obj
    # print 'max sol idx = %s' % str(sol2.idx)
    # print '# groups: ', len(sol2.groups)
    print 'time: %.4f sec' % sol2.running_time
    sum_t3 += sol2.running_time

    print '=== ratio %.3f (%.3f w.r. smp.) ===' % (sol2.ar, sol2.obj / sol0.obj)
    sum_r1 += sol2.ar;
    sum_r2 += sol2.obj / sol0.obj
    print ' avg r = %.3f (%.3f)' % (sum_r1 / t, sum_r2 / t)
    print ' avg t = g: %.3f | opt-s: %.3f | opt: %.3f' % (sum_t3 / t, sum_t1 / t, sum_t2 / t)

    sol3 = o.min_loss_OPF(ins, sol0.x, cons=cons)
    print 'OPT loss    = ', sol3.obj
    sol4 = o.min_loss_OPF(ins, sol2.x, cons=cons)
    print 'greedy loss = ', sol4.obj
    print

    if sol4.obj == - np.inf or sol3.obj == - np.inf:
        break
print 'step ', t
