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





for t in range(100000):
    T = i.network_38node(loss_ratio=.05)
    ins = i.rnd_instance_from_graph(T, n=100)
    # u.print_instance(ins) # print '----- OPT smp ------'
    ins.F = ins.I
    ins.I = []
    sol0 = s.OPT(ins, cons='', capacity_flag='C_')
    # print 'obj value: ', sol0.obj
    # print 'obj idx: ', sol0.idx
    # print 'time: %.4f sec'%sol0.running_time

    # print '-------- OPT -------'
    sol1 = o.max_OPF(ins, cons='')
    # print 'obj value: ', sol1.obj
    # print 'obj idx: ', sol1.idx
    # print 'time: %.4f sec'%sol1.running_time


    # print '----- greedy ------'
    ins.I = ins.F
    ins.F = []
    sol2 = s.greedy(ins, cons='', capacity_flag='C_')
    sol2.ar = sol2.obj / sol1.obj
    # print "max value =",  sol2.obj
    # print 'max sol idx = %s' % str(sol2.idx)
    # print '# groups: ', len(sol2.groups)
    # print 'time: %.4f sec'%sol2.running_time

    # print '\n=== ratio %.3f ==='%sol2.ar

    sol3 = o.min_loss_OPF(ins, sol0.idx)
    print 'OPT loss    = ', sol3.obj
    sol4 = o.min_loss_OPF(ins, sol2.idx)
    print 'greedy loss = ', sol4.obj

    if sol4.obj == - np.inf or sol3.obj == - np.inf:
        break
print 'step ', t
