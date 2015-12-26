# encoding=utf8
__author__ = 'mkhonji'

import numpy as np
import time, pickle
import util as u
import instance as ii
import s_maxOPF_algs as ss
import OPF_algs as oo


def sim(scenario="FCR", F_percentage=0, max_n=1500, step_n=100, start_n=100, reps=40, dry_run=False,
        dump_dir="results/dump/"):
    name = "%s_F_percentage=%.2f_max_n=%d_step_n=%d_start_n=%d_reps=%d" % (
        scenario, F_percentage, max_n, step_n, start_n, reps)
    ### set up variables
    cons = ''
    capacity_flag = 'C_'
    loss_ratio = .08
    t1 = time.time()
    #####

    assert (max_n % step_n == 0)

    greedy_time = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    greedy_ar = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    greedy_obj = np.zeros(((max_n - start_n + step_n) / step_n, reps))

    greedy_group_count = np.zeros(((max_n - start_n + step_n) / step_n, reps))

    OPT_s_time = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    OPT_s_obj = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    OPT_time = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    OPT_obj = np.zeros(((max_n - start_n + step_n) / step_n, reps))

    count = len(np.arange(start_n, max_n + 1, step_n))
    dump_data = {s: None for s in zip(np.arange(start_n, max_n + 1, step_n), np.arange(count))}

    for n in range(start_n, max_n + 1, step_n):
        for i in range(reps):
            print "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            print name
            print u"├── n=%d\n├── rep=%d\n└── elapsed time %f sec" % (n, i + 1, time.time() - t1)
            print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            T = ii.network_38node(loss_ratio=loss_ratio)
            ins = ii.sim_instance(T, scenario=scenario, n=n, F_percentage=F_percentage)
            sol_opt = oo.max_OPF(ins, cons)
            OPT_time[(n - start_n + step_n) / step_n - 1, i] = sol_opt.running_time
            OPT_obj[(n - start_n + step_n) / step_n - 1, i] = sol_opt.obj
            print "OPT obj   :    %15d  |  time: %5.3f" % (
                sol_opt.obj, sol_opt.running_time)
            sol_opt_s = ss.OPT(ins)
            OPT_s_time[(n - start_n + step_n) / step_n - 1, i] = sol_opt_s.running_time
            OPT_s_obj[(n - start_n + step_n) / step_n - 1, i] = sol_opt_s.obj
            print "OPTs obj  :    %15d  |  time: %5.3f" % (
                sol_opt_s.obj, sol_opt_s.running_time)
            sol = None
            if F_percentage == 0:
                sol = ss.greedy(ins, capacity_flag=capacity_flag, cons=cons)
            else:
                sol = ss.mixed_greedy(ins, capacity_flag=capacity_flag, cons=cons)
            sol.ar = sol.obj / sol_opt.obj

            greedy_time[(n - start_n + step_n) / step_n - 1, i] = sol.running_time
            greedy_obj[(n - start_n + step_n) / step_n - 1, i] = sol.obj
            greedy_ar[(n - start_n + step_n) / step_n - 1, i] = sol.ar
            greedy_group_count[(n - start_n + step_n) / step_n - 1, i] = len(sol.groups)
            print "Greedy obj:    %15d  |  time: %5.3f  |  AR: %5.3f (%d groups)" % (
                sol.obj, sol.running_time, sol.ar, len(sol.groups))
    if dry_run == False:
        np.savez(dump_dir + name,
                 greedy_obj=greedy_obj,
                 greedy_ar=greedy_ar,
                 greedy_time=greedy_time,
                 greedy_group_count=greedy_group_count,
                 OPT_obj=OPT_obj,
                 OPT_time=OPT_time,
                 OPT_s_obj=OPT_s_obj,
                 OPT_s_time=OPT_s_time)
    x = np.arange(start_n, max_n + 1, step_n)
    x = x.reshape((len(x), 1))

    mean_yerr_greedy_time = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_time)), 1)
    mean_yerr_OPT_time = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_time)), 1)
    mean_yerr_OPT_s_time = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_time)), 1)
    mean_yerr_greedy_ar = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_ar)), 1)
    mean_yerr_greedy_group_count = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_group_count)), 1)

    mean_yerr_greedy_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_obj)), 1)
    mean_yerr_OPT_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_obj)), 1)
    mean_yerr_OPT_s_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_obj)), 1)

    print mean_yerr_greedy_ar

    if dry_run == False:
        np.savez(dump_dir + name,
                 greedy_obj=greedy_obj,
                 greedy_ar=greedy_ar,
                 greedy_time=greedy_time,
                 greedy_group_count=greedy_group_count,
                 OPT_obj=OPT_obj,
                 OPT_time=OPT_time,
                 OPT_s_obj=OPT_s_obj,
                 OPT_s_time=OPT_s_time,
                 mean_yerr_greedy_ar=mean_yerr_greedy_ar,
                 mean_yerr_greedy_time=mean_yerr_greedy_time,
                 mean_yerr_greedy_group_count=mean_yerr_greedy_group_count,
                 mean_yerr_OPT_time=mean_yerr_OPT_time,
                 mean_yerr_OPT_s_time=mean_yerr_OPT_s_time,
                 mean_yerr_greedy_obj=mean_yerr_greedy_obj,
                 mean_yerr_OPT_obj=mean_yerr_OPT_obj,
                 mean_yerr_OPT_s_obj=mean_yerr_OPT_s_obj)
    fin_time = time.time() - t1
    m, s = divmod(fin_time, 60)
    h, m = divmod(m, 60)
    print "\n=== simulation finished in %d:%02d:%02d ===\n" % (h, m, s)

    return name


if __name__ == "__main__":
    sim(scenario="FUR", F_percentage=0.0, max_n=1500, step_n=100, start_n=100, reps=20)
