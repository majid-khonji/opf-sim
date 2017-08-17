# encoding=utf8
__author__ = 'Majid Khonji'

import numpy as np
import time, pickle
import util as u
import instance as ii
import s_maxOPF_algs as ss
import OPF_algs as oo

# topology = [38 | 123] (IEEE topology. The code can be cleaned up in future, only csv file should be given)
def sim(scenario="FCR", F_percentage=0.0, max_n=1500, step_n=100, start_n=100, reps=40, dry_run=False,
        dump_dir="results/dump/", is_adaptive_loss = True, topology=123):
    name = "%s__topology=%d__F_percentage=%.2f_max_n=%d_step_n=%d_start_n=%d_reps=%d" % (
        scenario, topology, F_percentage, max_n, step_n, start_n, reps)
    if is_adaptive_loss:
        name = "adapt__%s__topology=%d__F_percentage=%.2f_max_n=%d_step_n=%d_start_n=%d_reps=%d" % (
            scenario, topology, F_percentage, max_n, step_n, start_n, reps)
    ### set up variables
    cons = ''
    capacity_flag = 'C_'
    loss_ratio = .08
    loss_step = .005
    if is_adaptive_loss: loss_ratio = 0
    t1 = time.time()
    #####

    assert (max_n % step_n == 0)

    greedy_obj = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    greedy_time = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    greedy_ar = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    greedy_ar2 = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    adaptive_greedy_loss_ratio = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    adaptive_OPT_s_loss_ratio = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    sub_optimal1_count_per_iteration = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    sub_optimal2_count_per_iteration = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    opt_err_count_per_iteration = np.zeros(((max_n - start_n + step_n) / step_n, reps))

    greedy_group_count = np.zeros(((max_n - start_n + step_n) / step_n, reps))

    OPT_s_time = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    OPT_s_obj = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    OPT_time = np.zeros(((max_n - start_n + step_n) / step_n, reps))
    OPT_obj = np.zeros(((max_n - start_n + step_n) / step_n, reps))


    for n in range(start_n, max_n + 1, step_n):
        for i in range(reps):
            elapsed_time = time.time() - t1
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            print "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            print name
            print u"├── n=%d\n├── rep=%d\n└── elapsed time %d:%02d:%02d \n" % (n, i + 1, h, m, s)
            print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            T = None
            if topology == 38:
                # T = ii.network_38node(loss_ratio=loss_ratio)
                pass
            elif topology == 123:
                T = ii.network_csv_load(filename='test-feeders/123-node-line-data.csv',loss_ratio=loss_ratio)

            is_OPT_smaller = True

            suboptimal1_count = 0
            suboptimal2_count = 0
            opt_err_count = 0
            while is_OPT_smaller:
                ins = ii.sim_instance(T, scenario=scenario, n=n, F_percentage=F_percentage,capacity_flag=capacity_flag)
                ### opt
                sol_opt = oo.max_OPF_OPT(ins, cons)
                print "OPT obj   :    %15.2f  |  time: %5.3f" % (
                    sol_opt.obj, sol_opt.running_time)
                ### opt(s)
                sol_opt_s = None
                if is_adaptive_loss:
                    sol_opt_s = ss.adaptive_OPT(ins, cons=cons)
                else:
                    sol_opt_s = ss.OPT(ins, cons=cons,capacity_flag=capacity_flag)


                ####### replace if OPTs is bigger
                if sol_opt_s.obj > sol_opt.obj:
                    sol_opt.obj = sol_opt_s.obj
                    sol_opt.x = sol_opt_s.x

                OPT_time[(n - start_n + step_n) / step_n - 1, i] = sol_opt.running_time
                OPT_obj[(n - start_n + step_n) / step_n - 1, i] = sol_opt.obj
                OPT_s_time[(n - start_n + step_n) / step_n - 1, i] = sol_opt_s.running_time
                OPT_s_obj[(n - start_n + step_n) / step_n - 1, i] = sol_opt_s.obj

                print "OPTs obj  :    %15.2f  |  time: %5.3f" % (
                    sol_opt_s.obj, sol_opt_s.running_time)




                ### greedy
                sol = None
                if is_adaptive_loss:
                    sol = ss.adaptive_greedy(ins,cons=cons, loss_step = loss_step)
                elif F_percentage == 0:
                    sol = ss.greedy(ins, capacity_flag=capacity_flag, cons=cons)
                else:
                    sol = ss.mixed_greedy(ins, capacity_flag=capacity_flag, cons=cons)
                sol.ar = sol.obj / sol_opt.obj
                sol.ar2 = sol.obj / sol_opt_s.obj
                print "Greedy obj:    %15.2f  |  time: %5.3f  |  AR (OPT, OPTs): %5.3f, %4.3f (%d groups)" % (
                    sol.obj, sol.running_time, sol.ar, sol.ar2, len(sol.groups))

                if sol.ar > 1 or sol.ar2 >1: # or sol_opt_s.obj/sol_opt.obj > 1  :
                    if sol.ar > 1: suboptimal1_count += 1
                    if sol.ar2 > 1: suboptimal2_count += 1
                    #if sol_opt_s.obj/sol_opt.obj > 1: opt_err_count += 1
                    print '\n----- repeating (#Grd>OPT = %d, #Grd>OPTs = %d, #OPTs>OPT = %d): invalid OPT --------'%(suboptimal1_count, suboptimal2_count,opt_err_count)
                    print name
                    print u"├── n=%d\n├── rep=%d\n└── elapsed time %d:%02d:%02d \n" % (n, i + 1, h, m, s)

                    continue
                else: is_OPT_smaller = False

                greedy_obj[(n - start_n + step_n) / step_n - 1, i] = sol.obj
                greedy_time[(n - start_n + step_n) / step_n - 1, i] = sol.running_time
                greedy_ar[(n - start_n + step_n) / step_n - 1, i] = sol.ar
                greedy_ar2[(n - start_n + step_n) / step_n - 1, i] = sol.ar2
                greedy_group_count[(n - start_n + step_n) / step_n - 1, i] = len(sol.groups)
                sub_optimal1_count_per_iteration[(n - start_n + step_n) / step_n - 1, i] = suboptimal1_count
                sub_optimal2_count_per_iteration[(n - start_n + step_n) / step_n - 1, i] = suboptimal2_count
                opt_err_count_per_iteration[(n - start_n + step_n) / step_n - 1, i] = opt_err_count
                if is_adaptive_loss:
                    adaptive_greedy_loss_ratio[(n - start_n + step_n) / step_n - 1, i] = sol.loss_ratio
                    adaptive_OPT_s_loss_ratio[(n - start_n + step_n) / step_n - 1, i] = sol_opt_s.loss_ratio
        # intermediate saving
        if dry_run == False:
            np.savez(dump_dir + name,
                     greedy_obj=greedy_obj,
                     greedy_ar=greedy_ar,
                     greedy_ar2=greedy_ar2,
                     greedy_time=greedy_time,
                     greedy_group_count=greedy_group_count,
                     sub_optimal1_count_per_iteration=sub_optimal1_count_per_iteration,
                     sub_optimal2_count_per_iteration=sub_optimal2_count_per_iteration,
                     opt_err_count_per_iteration=opt_err_count_per_iteration,
                     adaptive_greedy_loss_ratio=adaptive_greedy_loss_ratio,
                     adaptive_OPT_s_loss_ratio=adaptive_OPT_s_loss_ratio,
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
    mean_yerr_greedy_ar2 = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_ar2)), 1)
    mean_yerr_greedy_group_count = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_group_count)), 1)

    mean_yerr_greedy_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_obj)), 1)
    mean_yerr_OPT_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_obj)), 1)
    mean_yerr_OPT_s_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_obj)), 1)

    print mean_yerr_greedy_ar

    if dry_run == False:
        np.savez(dump_dir + name,
                 greedy_obj=greedy_obj,
                 greedy_ar=greedy_ar,
                 greedy_ar2=greedy_ar2,
                 greedy_time=greedy_time,
                 greedy_group_count=greedy_group_count,
                 sub_optimal1_count_per_iteration=sub_optimal1_count_per_iteration,
                 sub_optimal2_count_per_iteration=sub_optimal2_count_per_iteration,
                 opt_err_count_per_iteration=opt_err_count_per_iteration,
                 adaptive_greedy_loss_ratio=adaptive_greedy_loss_ratio,
                 adaptive_OPT_s_loss_ratio=adaptive_OPT_s_loss_ratio,
                 OPT_obj=OPT_obj,
                 OPT_time=OPT_time,
                 OPT_s_obj=OPT_s_obj,
                 OPT_s_time=OPT_s_time,
                 mean_yerr_greedy_ar=mean_yerr_greedy_ar,
                 mean_yerr_greedy_ar2=mean_yerr_greedy_ar2,
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
    # import logging
    # logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    sim(scenario="FCM", F_percentage=0.75, max_n=1500, step_n=100, start_n=100, reps=50)





    
    

   
    
