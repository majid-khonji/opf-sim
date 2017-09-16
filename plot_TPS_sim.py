__author__ = 'Majid Khonji'
import matplotlib.pyplot as plt
import numpy as np
import util as u

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


def quick_plot(name, dump_dir="results/dump/", fig_dir="results/"):
    plt.ioff()
    plt.clf()
    plt.figure(figsize=(5, 3))
    f = np.load(dump_dir + '/' + name + ".npz")

    x = np.arange(100, 1500 + 1, 100)
    x = x.reshape((len(x), 1))
    print f.files
    greedy_obj = f["greedy_obj"]
    OPT_obj = f["OPT_obj"]
    OPT_s_obj = f["OPT_s_obj"]
    mean_yerr_greedy_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_obj)), 1)
    print mean_yerr_greedy_obj
    mean_yerr_OPT_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_obj)), 1)
    mean_yerr_OPT_s_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_obj)), 1)


    plt.errorbar(mean_yerr_greedy_obj[:, 0], mean_yerr_greedy_obj[:, 1],
                 yerr=mean_yerr_greedy_obj[:, 2], linestyle='None', color='blue')
    plt.plot(mean_yerr_greedy_obj[:, 0], mean_yerr_greedy_obj[:, 1], color='blue', label="GR", linewidth=2,
             linestyle='-.')

    plt.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1],
                 yerr=mean_yerr_OPT_obj[:, 2], linestyle='None', color='red')
    plt.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='red', label="OPT", linewidth=2,
             linestyle='-.')

    plt.errorbar(mean_yerr_OPT_s_obj[:, 0], mean_yerr_OPT_s_obj[:, 1],
                 yerr=mean_yerr_OPT_s_obj[:, 2], linestyle='None', color='green')
    plt.plot(mean_yerr_OPT_s_obj[:, 0], mean_yerr_OPT_s_obj[:, 1], color='green', label="sOPT", linewidth=2,
             linestyle='-.')

    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plt.legend()
    # plt.ylim([0,1])
    plt.xlabel("Number of customers")
    plt.ylabel("Objective value")
    plt.subplots_adjust(left=0.125, bottom=0.15, right=.9, top=.85)
    plt.savefig(fig_dir + '/quick.pdf')


def _format_exponent(ax, axis='y', y_horz_alignment='left'):
    # Change the ticklabel format to scientific format
    # ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(0, 4))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        if y_horz_alignment == 'right':
            x_pos = 1
        y_pos = 1.0
        horizontalalignment = y_horz_alignment
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' % expo
        if y_horz_alignment == 'right':
            offset_text = r'$\mathregular{10^{%d}}$x' % expo

        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)

        # Add in a text box at the top of the y axis
        ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment)
    return ax

# modified (takes the max for OPT)
def plot_subfig_obj(name, ax, dump_dir, start_n=100, max_n=1500, step_n=100):

    f = np.load(dump_dir + '/' + name + ".npz")

    print "plotting %s..."%name
    round_OPF_obj = f['round_OPF_obj']
    OPT_obj = f['OPT_obj']
    frac_OPF_obj = f['frac_OPF_obj']#np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_obj)), 1)

    x = np.arange(start_n, max_n +1 , step_n)
    x = x.reshape((len(x), 1))

    print round_OPF_obj.shape
    print x.shape

    mean_yerr_round_OPF_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_obj)), 1)
    mean_yerr_frac_OPF_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), frac_OPF_obj)), 1)
    mean_yerr_OPT_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_obj)), 1)


    # mean_yerr_round_OPF_obj = f['mean_yerr_round_OPF_obj']
    # mean_yerr_OPT_obj = f['mean_yerr_OPT_obj']
    # mean_yerr_frac_OPF_obj = f['mean_yerr_frac_OPF_obj']#np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_obj)), 1)

    ax.errorbar(mean_yerr_round_OPF_obj[:, 0], mean_yerr_round_OPF_obj[:, 1],
                yerr=mean_yerr_round_OPF_obj[:, 2], linestyle='None', color='blue')
    ax.plot(mean_yerr_round_OPF_obj[:, 0], mean_yerr_round_OPF_obj[:, 1], color='blue', marker='.', label="Alg.",
            linewidth=2,
            linestyle='-.')


    # ax.errorbar(mean_yerr_frac_OPF_obj[:, 0], mean_yerr_frac_OPF_obj[:, 1],
    #             yerr=mean_yerr_frac_OPF_obj[:, 2], linestyle='None', color='green')
    # ax.plot(mean_yerr_frac_OPF_obj[:, 0], mean_yerr_frac_OPF_obj[:, 1], color='green', label=r"frac",
    #         linewidth=2, linestyle='--')

    ax.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], yerr=mean_yerr_OPT_obj[:, 2], color='red',
                linestyle='None')
    ax.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='red', label=r'OPT', linewidth=2)


    ax.grid(True)

    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([start_n, max_n])

def plot_subfig_ar(name, ax, dump_dir, start_n=100,step_n=100, max_n=1500):

    f = np.load(dump_dir + '/' + name + ".npz")
    round_OPF_ar = f['round_OPF_ar']
    # OPT_ar = f['OPT_ar']
    # frac_OPF_ar = f['frac_OPF_ar']

    x = np.arange(start_n, max_n + 1 , step_n)
    x = x.reshape((len(x), 1))
    print name
    print start_n, step_n, max_n
    print round_OPF_ar.shape
    print x.shape

    mean_yerr_round_OPF_ar = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_ar)), 1)
    # mean_yerr_frac_OPF_ar = np.append(x, np.array(map(lambda y: u.mean_yerr(y), frac_OPF_ar)), 1)

    # mean_yerr_round_OPF_ar = f['mean_yerr_round_OPF_ar']
    # mean_yerr_no_LP_round_OPF_ar = f['mean_yerr_round_OPF_ar']

    ax.errorbar(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1],
                yerr=mean_yerr_round_OPF_ar[:, 2], linestyle='None', color='blue')
    ax.plot(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1], color='blue', marker='.', label="round",
            linewidth=2,
            linestyle='-.')


    # ax.errorbar(mean_yerr_no_LP_round_OPF_ar[:, 0], mean_yerr_no_LP_round_OPF_ar[:, 1],
    #             yerr=mean_yerr_no_LP_round_OPF_ar[:, 2], linestyle='None', color='green')
    # ax.plot(mean_yerr_no_LP_round_OPF_ar[:, 0], mean_yerr_no_LP_round_OPF_ar[:, 1], color='green', label=r"round_no_lp",
    #         linewidth=2, linestyle='--')

    ax.grid(True)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([start_n, max_n])



def plot_subfig_time(name, ax, dump_dir):

    f = np.load(dump_dir + '/' + name + ".npz")
    mean_yerr_greedy_time = f['mean_yerr_greedy_time']
    # mean_yerr_greedy_time[:,1:] = 1000 * mean_yerr_greedy_time[:,1:]
    mean_yerr_OPT_time = f['mean_yerr_OPT_time']
    # mean_yerr_OPT_time[:,1:] = 1000 * mean_yerr_OPT_time[:,1:]
    mean_yerr_OPT_s_time = f['mean_yerr_OPT_s_time']

    ax.errorbar(mean_yerr_greedy_time[:, 0], mean_yerr_greedy_time[:, 1],
                yerr=mean_yerr_greedy_time[:, 2], linestyle='None', color='blue')
    ax.plot(mean_yerr_greedy_time[:, 0], mean_yerr_greedy_time[:, 1], color='blue', marker='.', label="Alg.2",
            linewidth=2,
            linestyle='-.')

    # c = ax.twinx()

    ax.errorbar(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1], yerr=mean_yerr_OPT_time[:, 2], color='red',
                linestyle='None')
    ax.plot(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1], color='red', label=r'OPT', linewidth=2)


    # c.errorbar(mean_yerr_OPT_s_time[:, 0], mean_yerr_OPT_s_time[:, 1],
    #             yerr=mean_yerr_OPT_s_time[:, 2], linestyle='None', color='green')
    # c.plot(mean_yerr_OPT_s_time[:, 0], mean_yerr_OPT_s_time[:, 1], color='green', label=r"OPT(s)",
    #         linewidth=2, linestyle='--')
    # ax.set_ylim([0,.900])
    ax.grid(True)
    # c.grid(True)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # format_exponent(ax, 'y')
    # format_exponent(c, 'y', y_horz_alignment='right')

    # for tick in c.get_yticklabels():
    #     tick.set_color('red')
    # for tick in ax.get_yticklabels():
    #     tick.set_color('blue')

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([100, 2050])

    # return c
def plot_all_obj(dump_dir="results/dump/", network=123, fig_dir="results/"):
    """
    :param dump_dir:
    :param fig_dir:
    :return:
    """
    plt.ioff()
    # plt.clf()
    FCR_name = ''
    FCM_name = ''
    FUR_name = ''
    FUM_name = ''

    if network == 123:
        FCR_name = 'TPS:[FCR]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TPS:[FCM]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TPS:[FUR]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TPS:[FUM]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
    elif network==13:
        FCR_name = 'TPS:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FCM_name = 'TPS:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUR_name = 'TPS:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUM_name = 'TPS:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
    else:
        print "wrong network"
        return


    # fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 5))

    fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)


    if network ==13:
        plot_subfig_obj(name=FCR_name, ax=fcr, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fcr.set_title('CR')
        fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        plot_subfig_obj(name=FCM_name, ax=fcm, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fcm.set_title('CM')
        plot_subfig_obj(name=FUR_name, ax=fur, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fur.set_title('UR')
        plot_subfig_obj(name=FUM_name, ax=fum, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fum.set_title('UM')
    else:
        plot_subfig_obj(name=FCR_name, ax=fcr, dump_dir=dump_dir)
        fcr.set_title('CR')
        fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        plot_subfig_obj(name=FCM_name, ax=fcm, dump_dir=dump_dir)
        fcm.set_title('CM')
        plot_subfig_obj(name=FUR_name, ax=fur, dump_dir=dump_dir)
        fur.set_title('UR')
        plot_subfig_obj(name=FUM_name, ax=fum, dump_dir=dump_dir)
        fum.set_title('UM')
        # fcm.set_ylim([0,7*10**12])


    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Objective value', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
    if network == 123:
        plt.savefig(fig_dir + "TPS_obj_123.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir + "TPS_obj_13.pdf", bbox_inches='tight')



def plot_all_ar(dump_dir="results/dump/",network=13, fig_dir="results/", ar_type="opt"):
    """
    :param dump_dir:
    :param fig_dir:
    :param type: "opt"|"opt(s)"  (calculate approximation ratio vs opt or opt(s)
    :return:
    """
    plt.ioff()
    plt.clf()
    if network == 123:
        FCR_name = 'TPS:[FCR]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TPS:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUR_name = 'TPS:[FUR]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TPS:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
    elif network==13:
        FCR_name = 'TPS:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FCM_name = 'TPS:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUR_name = 'TPS:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUM_name = 'TPS:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
    else:
        print "wrong network"
        return

    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 5))

    if network ==13:
        plot_subfig_ar(name=FCR_name, ax=fcr, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fcr.set_title('CR')
        fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        plot_subfig_ar(name=FCM_name, ax=fcm, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fcm.set_title('CM')
        plot_subfig_ar(name=FUR_name, ax=fur, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fur.set_title('UR')
        plot_subfig_ar(name=FUM_name, ax=fum, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fum.set_title('UM')
    else:
        plot_subfig_ar(name=FCR_name, ax=fcr, dump_dir=dump_dir)
        fcr.set_title('CR')
        fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        plot_subfig_ar(name=FCM_name, ax=fcm, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fcm.set_title('CM')
        plot_subfig_ar(name=FUR_name, ax=fur, dump_dir=dump_dir)
        fur.set_title('UR')
        plot_subfig_ar(name=FUM_name, ax=fum, dump_dir=dump_dir, max_n = 3500, start_n=2000, step_n=100)
        fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, 'Approximation ratio', ha='center', va='center', rotation='vertical', fontsize=14)
    if network == 13:
        fcr.set_ylim([.0, 10.5])
        fcm.set_ylim([.0, 10.5])
        fur.set_ylim([.0, 10.5])
        fum.set_ylim([.0, 10.5])
        plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
        plt.savefig(fig_dir +"TPS_ar_13.pdf", bbox_inches='tight')
        print 'saving'
    elif network == 123:
        fcr.set_ylim([.9, 1.2])
        fcm.set_ylim([.9, 1.2])
        fur.set_ylim([.9, 1.2])
        fum.set_ylim([.9, 1.2])
        plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
        plt.savefig(fig_dir +"TPS_ar_123.pdf", bbox_inches='tight')

def plot_all_time(dump_dir="results/dump/", fig_dir="results/"):
    """
    :param dump_dir:
    :param fig_dir:
    :return:
    """
    plt.ioff()
    # plt.clf()
    FCR_name = "FCR__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    FCM_name = "FCM__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    FUR_name = "FUR__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    FUM_name = "FUM__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    # FCR_name = "FCR_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    # FCM_name = "FCM_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    # FUR_name = "FUR_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    # FUM_name = "FUM_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"

    # fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 5))

    c=plot_subfig_time(name=FCR_name, ax=fcr, dump_dir=dump_dir)
    fcr.set_title('CR')
    fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
    # c.legend(bbox_to_anchor=(.75, 1.15, 0, 0), loc=4, ncol=1, borderaxespad=0., fontsize=12)
    #fcr.set_ylim([0,120])
    # c.set_ylim([0,80000])

    c= plot_subfig_time(name=FCM_name, ax=fcm, dump_dir=dump_dir)
    fcm.set_title('CM')
    # fcm.set_ylim([0,1])
    #fcm.set_ylim([0,8.1])

    c = plot_subfig_time(name=FUR_name, ax=fur, dump_dir=dump_dir)
    fur.set_title('UR')
    # fur.set_ylim([0,1])
    #fur.set_ylim([0,5])

    c = plot_subfig_time(name=FUM_name, ax=fum, dump_dir=dump_dir)
    fum.set_title('UM')
    # fum.set_ylim([0,1])
    #fum.set_ylim([0,10])

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Running Time (seconds)', ha='center', va='center', rotation='vertical', fontsize=14)
    # fig.text(-0.01, 0.5, 'Running time (milliseconds)', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
    plt.savefig(fig_dir + "time_123_all.pdf", bbox_inches='tight')

def plot_time(dump_dir="results/dump/", fig_dir="results/"):
    # plt.clf()
    FCR_name = "slow_FCR_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    FCM_name = "FCM_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    FUR_name = "FUR_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    FUM_name = "FUM_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    max_n = 2000;
    step_n = 100;
    start_n = 100;

    f = np.load(dump_dir + '/' + FCR_name + ".npz")
    greedy_FCR = f["greedy_time"]
    OPT_FCR = f["OPT_time"]
    f = np.load(dump_dir + '/' + FCM_name + ".npz")
    greedy_FCM = f["greedy_time"]
    OPT_FCM = f["OPT_time"]
    f = np.load(dump_dir + '/' + FUR_name + ".npz")
    greedy_FUR = f["greedy_time"]
    OPT_FUR = f["OPT_time"]
    f = np.load(dump_dir + '/' + FUM_name + ".npz")
    greedy_FUM = f["greedy_time"]
    OPT_FUM = f["OPT_time"]

    gr_time = 1000 * np.append(np.append(greedy_FCR, greedy_FCM, 1),
                               np.append(greedy_FUR, greedy_FUM, 1), 1)
    OPT_time = 1000 * np.append(np.append(OPT_FCR, OPT_FCM, 1), np.append(OPT_FUR, OPT_FUM, 1), 1)

    x = np.arange(start_n, max_n + 1, step_n)
    x = x.reshape((len(x), 1))
    mean_yerr_gr_time = np.append(x, np.array(map(lambda y: u.mean_yerr(y), gr_time)), 1)
    mean_yerr_OPT_time = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_time)), 1)

    fig = plt.figure(figsize=(7, 3))
    g = plt.subplot(121)
    g.errorbar(mean_yerr_gr_time[:, 0], mean_yerr_gr_time[:, 1],
               yerr=mean_yerr_gr_time[:, 2], linestyle='None', color='blue')
    g.plot(mean_yerr_gr_time[:, 0], mean_yerr_gr_time[:, 1], color='blue', label="Alg.2", linewidth=2,
           linestyle='-', marker='.')
    plt.xticks(rotation=35)
    plt.legend(loc=2)
    # plt.ylabel("Running time", fontsize=14)
    plt.ylim([0, 500])
    plt.xlim([100, 2050])
    g.grid(True)
    _format_exponent(g, 'y')

    o = plt.subplot(122)
    o.errorbar(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1],
               yerr=mean_yerr_OPT_time[:, 2], linestyle='None', color='red')
    o.plot(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1], color='red', label="OPT", linewidth=2,
           linestyle='-')
    plt.xticks(rotation=35)
    plt.legend(loc=2)
    plt.ylim([0, 25000])
    plt.xlim([100, 2050])
    o.grid(True)
    _format_exponent(o, 'y')

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)
    fig.text(-0.01, 0.5, 'Running time (milliseconds)', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
    plt.savefig(fig_dir + "time.pdf", bbox_inches='tight')


if __name__ == "__main__":
    # quick_plot(name="FCM_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=20")
    # plot_all_obj()
    plot_all_ar(ar_type="opt")
    plot_all_time()
    # plot_all_loss()
    # plot_time()
