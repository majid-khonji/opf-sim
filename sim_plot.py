__author__ = 'mkhonji'
import matplotlib.pyplot as plt
import numpy as np
import ckp_sim as s
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


def plot_results(name, dump_dir="results/dump/", fig_dir="results/"):
    plt.ioff()
    plt.clf()
    plt.figure(figsize=(5, 3))
    f = np.load(dump_dir + '/' + name + ".npz")

    mean_yerr_greedy_ratio_obj = f["mean_yerr_greedy_ratio_obj"]

    plt.errorbar(mean_yerr_greedy_ratio_obj[:, 0], mean_yerr_greedy_ratio_obj[:, 1],
                 yerr=mean_yerr_greedy_ratio_obj[:, 2], linestyle='None', color='blue')
    plt.plot(mean_yerr_greedy_ratio_obj[:, 0], mean_yerr_greedy_ratio_obj[:, 1], color='blue', label="GRA", linewidth=2,
             linestyle='-.')

    mean_yerr_greedy_demand_obj = f["mean_yerr_greedy_demand_obj"]
    plt.errorbar(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1],
                 yerr=mean_yerr_greedy_demand_obj[:, 2], linestyle='None', color='green')
    plt.plot(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1], color='green', label="GDA",
             linewidth=2, linestyle='--')

    mean_yerr_greedy_val_obj = f["mean_yerr_greedy_val_obj"]
    plt.errorbar(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1],
                 yerr=mean_yerr_greedy_val_obj[:, 2], linestyle='None', color='black')
    plt.plot(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1], color='black', label="GUA",
             linewidth=2, linestyle='--')

    mean_yerr_OPT_obj = f["mean_yerr_OPT_obj"]
    plt.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], yerr=mean_yerr_OPT_obj[:, 2], color='red',
                 linestyle='None')
    plt.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='red', label='OPT', linewidth=2)

    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plt.legend()
    # plt.ylim([1000000000,10000000000000])
    plt.xlabel("Number of customers")
    plt.ylabel("Maximized Valuation")
    plt.subplots_adjust(left=0.125, bottom=0.15, right=.9, top=.85)
    plt.savefig(fig_dir + '/' + name + '.pdf')


def format_exponent(ax, axis='y', y_horz_alignment='left'):
    # Change the ticklabel format to scientific format
    # ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(0, 0))

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


def plot_subfig(name, ax, dump_dir, fig_dir, type="obj"):
    extension = None
    if type == "obj":
        extension = "obj"
    elif type in ["cr", "cr_pj"]:
        extension = "cr"

    f = np.load(dump_dir + '/' + name + ".npz")
    mean_yerr_greedy_ratio_obj = f["mean_yerr_greedy_ratio_" + extension]

    ax.errorbar(mean_yerr_greedy_ratio_obj[:, 0], mean_yerr_greedy_ratio_obj[:, 1],
                yerr=mean_yerr_greedy_ratio_obj[:, 2], linestyle='None', color='blue')
    ax.plot(mean_yerr_greedy_ratio_obj[:, 0], mean_yerr_greedy_ratio_obj[:, 1], color='blue', marker='.', label="GRA",
            linewidth=2,
            linestyle='-.')

    if type not in ["cr", "cr_pj"]:
        mean_yerr_OPT_obj = f["mean_yerr_OPT_" + extension]
        ax.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], yerr=mean_yerr_OPT_obj[:, 2], color='red',
                    linestyle='None')
        ax.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='red', label='OPT', linewidth=2)
        mean_yerr_greedy_demand_obj = f["mean_yerr_greedy_demand_" + extension]
        ax.errorbar(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1],
                    yerr=mean_yerr_greedy_demand_obj[:, 2], linestyle='None', color='green')
        ax.plot(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1], color='green', label="GDA",
                linewidth=2, linestyle='--')

        mean_yerr_greedy_val_obj = f["mean_yerr_greedy_val_" + extension]
        ax.errorbar(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1],
                    yerr=mean_yerr_greedy_val_obj[:, 2], linestyle='None', color='black')
        ax.plot(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1], color='black', label="GUA",
                linewidth=2, linestyle='--')
    elif type == "cr_pj":
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        mean_yerr_OPT_obj = f["pj_mean_yerr_OPT_" + extension]
        ax.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], yerr=mean_yerr_OPT_obj[:, 2], color='red',
                    linestyle='None')
        ax.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='red', label=r'$\mathrm{OPT}_\mathrm{pj}$',
                linewidth=2)
        mean_yerr_greedy_demand_obj = f["pj_mean_yerr_greedy_demand_" + extension]
        ax.errorbar(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1],
                    yerr=mean_yerr_greedy_demand_obj[:, 2], linestyle='None', color='green')
        ax.plot(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1], color='green',
                label="$\mathrm{GDA}_\mathrm{pj}$",
                linewidth=2, linestyle='--')

        mean_yerr_greedy_val_obj = f["pj_mean_yerr_greedy_val_" + extension]
        ax.errorbar(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1],
                    yerr=mean_yerr_greedy_val_obj[:, 2], linestyle='None', color='black')
        ax.plot(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1], color='black',
                label="$\mathrm{GUA}_\mathrm{pj}$",
                linewidth=2, linestyle='--')
    elif type == "cr":
        mean_yerr_greedy_demand_obj = f["mean_yerr_greedy_demand_" + extension]
        ax.errorbar(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1],
                    yerr=mean_yerr_greedy_demand_obj[:, 2], linestyle='None', color='green')
        ax.plot(mean_yerr_greedy_demand_obj[:, 0], mean_yerr_greedy_demand_obj[:, 1], color='green', label="GDA",
                linewidth=2, linestyle='--')

        mean_yerr_greedy_val_obj = f["mean_yerr_greedy_val_" + extension]
        ax.errorbar(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1],
                    yerr=mean_yerr_greedy_val_obj[:, 2], linestyle='None', color='black')
        ax.plot(mean_yerr_greedy_val_obj[:, 0], mean_yerr_greedy_val_obj[:, 1], color='black', label="GUA",
                linewidth=2, linestyle='--')
    ax.grid(True)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    format_exponent(ax, 'y')
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([100, 1500])


def plot_all(dump_dir="results/dump/", fig_dir="results/", type="obj"):
    """

    :param dump_dir:
    :param fig_dir:
    :param type: obj/cr/cr_pj/time
    :return:
    """
    plt.ioff()

    # plt.clf()

    FCR_name = "ckp_sim:FCR_C=2000000_max_n=1500_step_n=100_start_n=100_reps=20"
    FCM_name = "ckp_sim:FCM_C=2000000_max_n=1500_step_n=100_start_n=100_reps=100"
    FUR_name = "ckp_sim:FUR_C=2000000_max_n=1500_step_n=100_start_n=100_reps=20"
    FUM_name = "ckp_sim:FUM_C=2000000_max_n=1500_step_n=100_start_n=100_reps=100"

    # fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 5))

    plot_subfig(name=FCR_name, ax=fcr, dump_dir=dump_dir, fig_dir=fig_dir, type=type)
    fcr.set_title('CR')

    fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    plot_subfig(name=FCM_name, ax=fcm, dump_dir=dump_dir, fig_dir=fig_dir, type=type)
    fcm.set_title('CM')

    plot_subfig(name=FUR_name, ax=fur, dump_dir=dump_dir, fig_dir=fig_dir, type=type)
    fur.set_title('UR')

    plot_subfig(name=FUM_name, ax=fum, dump_dir=dump_dir, fig_dir=fig_dir, type=type)
    fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)

    if (type == "obj"):
        fig.text(0.00, 0.5, 'Maximized utility', ha='center', va='center', rotation='vertical', fontsize=14)
    elif (type in ['cr', 'cr_pj']):
        fig.text(0.00, 0.5, 'Approximation ratio', ha='center', va='center', rotation='vertical', fontsize=14)
        fcr.set_ylim([0, 1.5])
        fcm.set_ylim([0, 1.5])
        fur.set_ylim([0, 1.5])
        fum.set_ylim([0, 1.5])

    plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
    plt.savefig(fig_dir + type + ".pdf", bbox_inches='tight')


def plot_time(dump_dir="results/dump/", fig_dir="results/"):
    # plt.clf()
    FCR_name = "ckp_sim:FCR_C=2000000_max_n=1500_step_n=100_start_n=100_reps=20"
    FCM_name = "ckp_sim:FCM_C=2000000_max_n=1500_step_n=100_start_n=100_reps=100"
    FUR_name = "ckp_sim:FUR_C=2000000_max_n=1500_step_n=100_start_n=100_reps=20"
    FUM_name = "ckp_sim:FUM_C=2000000_max_n=1500_step_n=100_start_n=100_reps=100"
    max_n = 1500;
    step_n = 100;
    start_n = 100;
    reps = 20

    f = np.load(dump_dir + '/' + FCR_name + ".npz")
    mean_yerr_greedy_ratio_FCR = f["mean_yerr_greedy_ratio_time"]
    f = np.load(dump_dir + '/' + FCM_name + ".npz")
    mean_yerr_greedy_ratio_FCM = f["mean_yerr_greedy_ratio_time"]
    f = np.load(dump_dir + '/' + FUR_name + ".npz")
    mean_yerr_greedy_ratio_FUR = f["mean_yerr_greedy_ratio_time"]
    f = np.load(dump_dir + '/' + FUM_name + ".npz")
    mean_yerr_greedy_ratio_FUM = f["mean_yerr_greedy_ratio_time"]

    f = np.load(dump_dir + '/' + FCR_name + ".npz")
    greedy_ratio_FCR = f["greedy_ratio_time"]
    OPT_FCR = f["OPT_time"]
    f = np.load(dump_dir + '/' + FCM_name + ".npz")
    greedy_ratio_FCM = f["greedy_ratio_time"]
    OPT_FCM = f["OPT_time"]
    f = np.load(dump_dir + '/' + FUR_name + ".npz")
    greedy_ratio_FUR = f["greedy_ratio_time"]
    OPT_FUR = f["OPT_time"]
    f = np.load(dump_dir + '/' + FUM_name + ".npz")
    greedy_ratio_FUM = f["greedy_ratio_time"]
    OPT_FUM = f["OPT_time"]

    gr_time = 1000 * np.append(np.append(greedy_ratio_FCR, greedy_ratio_FCM, 1),
                               np.append(greedy_ratio_FUR, greedy_ratio_FUM, 1), 1)
    OPT_time = 1000 * np.append(np.append(OPT_FCR, OPT_FCM, 1), np.append(OPT_FUR, OPT_FUM, 1), 1)

    x = np.arange(start_n, max_n + 1, step_n)
    x = x.reshape((len(x), 1))
    mean_yerr_gr_time = np.append(x, np.array(map(lambda y: s.mean_yerr(y), gr_time)), 1)
    mean_yerr_OPT_time = np.append(x, np.array(map(lambda y: s.mean_yerr(y), OPT_time)), 1)

    fig = plt.figure(figsize=(7, 3))
    g = plt.subplot(121)
    g.errorbar(mean_yerr_gr_time[:, 0], mean_yerr_gr_time[:, 1],
               yerr=mean_yerr_gr_time[:, 2], linestyle='None', color='blue')
    g.plot(mean_yerr_gr_time[:, 0], mean_yerr_gr_time[:, 1], color='blue', label="GRA", linewidth=2,
           linestyle='-', marker='.')
    plt.xticks(rotation=35)
    plt.legend(loc=2)
    # plt.ylabel("Running time", fontsize=14)
    plt.ylim([0, 40])
    plt.xlim([100, 1500])
    g.grid(True)
    format_exponent(g, 'y')

    o = plt.subplot(122)
    o.errorbar(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1],
               yerr=mean_yerr_OPT_time[:, 2], linestyle='None', color='red')
    o.plot(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1], color='red', label="OPT", linewidth=2,
           linestyle='-')
    plt.xticks(rotation=35)
    plt.legend(loc=2)
    plt.ylim([0, 40000])
    plt.xlim([100, 1500])
    o.grid(True)
    format_exponent(o, 'y')

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)
    fig.text(-0.01, 0.5, 'Running time (milliseconds)', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
    plt.savefig(fig_dir + "time.pdf", bbox_inches='tight')


def plot_dyn_C(dump_dir="results/dump/", fig_dir="results/"):
    plt.ioff()
    # plt.clf()
    FUM_name = "ckp_sim_dyn:FUM__C=2000000__T=10000__max_n=1000__reps=20__C_drop=(5,35)__failure_prob=0.65__avg_fail=200"
    FUR_name = "ckp_sim_dyn:FUR__C=2000000__T=10000__max_n=1000__reps=20__C_drop=(5,35)__failure_prob=0.65__avg_fail=200"
    FCM_name = "ckp_sim_dyn:FCM__C=2000000__T=10000__max_n=1000__reps=20__C_drop=(5,35)__failure_prob=0.65__avg_fail=200"

    f = np.load(dump_dir + '/' + FUM_name + ".npz")
    mean_yerr_greedy_ratio_FUM = f["mean_yerr_greedy_ratio_obj"]
    mean_yerr_OPT_FUM = f["mean_yerr_OPT_obj"]
    C_FUM = f["C_e"]

    f = np.load(dump_dir + '/' + FUR_name + ".npz")
    mean_yerr_greedy_ratio_FUR = f["mean_yerr_greedy_ratio_obj"]
    mean_yerr_OPT_FUR = f["mean_yerr_OPT_obj"]
    C_FUR = f["C_e"]




    ###############################
    fig, (g, g2) = plt.subplots(1, 2, sharex='col', figsize=(7, 5))

    # fig = plt.figure(figsize=(7,4), sharex='col')
    g = plt.subplot(211)

    x = mean_yerr_greedy_ratio_FUM[:, 0]
    g.errorbar(x, mean_yerr_greedy_ratio_FUM[:, 1],
               yerr=mean_yerr_greedy_ratio_FUM[:, 2], linestyle='None', color='blue')
    g.step(x, mean_yerr_greedy_ratio_FUM[:, 1], color='blue', label="GRA", linewidth=2,
           linestyle='-', marker='.')

    g.errorbar(x, mean_yerr_OPT_FUM[:, 1],
               yerr=mean_yerr_OPT_FUM[:, 2], linestyle='None', color='red')
    g.step(x, mean_yerr_OPT_FUM[:, 1], color='red', label="OPT", linewidth=2,
           linestyle='-')
    # plt.xticks(rotation=30)
    g.grid(True)
    g.set_title("UM")
    g.set_ylim([0, 10000000])

    c = g.twinx()
    c.step(x, C_FUM, color='green', label="Capacity", linewidth=1, linestyle='-')
    c.set_ylabel("              Capacity", color='green', fontsize=14)
    c.set_ylim([-2000000, 2500000])

    for tick in c.get_yticklabels():
        tick.set_color('green')
    c.get_yticklabels()[0].set_visible(False)
    c.get_yticklabels()[1].set_visible(False)
    c.get_yticklabels()[2].set_visible(False)
    c.get_yticklabels()[3].set_visible(False)

    format_exponent(g, 'y')
    format_exponent(c, 'y', y_horz_alignment='right')
    g.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=2, borderaxespad=0., fontsize=12)
    c.legend(bbox_to_anchor=(1., 1.15, 0, 0), loc=4, ncol=1, borderaxespad=0., fontsize=12)
    #############################
    #############################
    g2 = plt.subplot(212)
    x = mean_yerr_greedy_ratio_FUR[:, 0]

    g2.errorbar(x, mean_yerr_greedy_ratio_FUR[:, 1],
                yerr=mean_yerr_greedy_ratio_FUR[:, 2], linestyle='None', color='blue')
    g2.step(x, mean_yerr_greedy_ratio_FUR[:, 1], color='blue', label="GRA", linewidth=2,
            linestyle='-', marker='.')


    # print mean_yerr_OPT_FUR[:, 1]
    # print mean_yerr_greedy_ratio_FUR[:,1]
    g2.errorbar(x, mean_yerr_OPT_FUR[:, 1],
                yerr=mean_yerr_OPT_FUR[:, 2], linestyle='None', color='red')
    g2.step(x, mean_yerr_OPT_FUR[:, 1], color='red', label="OPT", linewidth=2,
            linestyle='-')
    # plt.xticks(rotation=30)
    g2.grid(True)
    g2.set_title("UR")
    g2.set_ylim([0, 5000000])

    c2 = g2.twinx()
    c2.step(x, C_FUR, color='green', label="Capacity", linewidth=1, linestyle='-')
    c2.set_ylabel("              Capacity", color='green', fontsize=14)
    c2.set_ylim([-2000000, 2500000])
    g2.set_xlim([0, 10000])

    for tick in c2.get_yticklabels():
        tick.set_color('green')
    c2.get_yticklabels()[0].set_visible(False)
    c2.get_yticklabels()[1].set_visible(False)
    c2.get_yticklabels()[2].set_visible(False)
    c2.get_yticklabels()[3].set_visible(False)
    format_exponent(g2, 'y')
    format_exponent(c2, 'y', y_horz_alignment='right')

    fig.text(0.5, 0.01, 'Time', ha='center', va='center', fontsize=14)
    fig.text(-0.01, 0.5, 'Maximized utility', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)

    plt.savefig(fig_dir + "dyn.pdf", bbox_inches='tight')
