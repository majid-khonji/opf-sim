__author__ = 'majid Khonji'
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
def plot_subfig_obj(name, ax, dump_dir):

    f = np.load(dump_dir + '/' + name + ".npz")
    greedy_obj = f["greedy_obj"]
    OPT_obj = f["OPT_obj"]
    OPT_s_obj = f["OPT_s_obj"]
    ############ new
    OPT_obj = np.maximum(OPT_s_obj, OPT_obj)
    x = np.arange(100, 1500 + 1, 100)
    x = x.reshape((len(x), 1))
    mean_yerr_greedy_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_obj)), 1)
    mean_yerr_OPT_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_obj)), 1)
    mean_yerr_OPT_s_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_obj)), 1)

    ax.errorbar(mean_yerr_greedy_obj[:, 0], mean_yerr_greedy_obj[:, 1],
                yerr=mean_yerr_greedy_obj[:, 2], linestyle='None', color='blue',capsize=2,elinewidth=.8)
    ax.plot(mean_yerr_greedy_obj[:, 0], mean_yerr_greedy_obj[:, 1], color='blue', marker='.', label="GreedyOPF",
            linewidth=2)


    ax.errorbar(mean_yerr_OPT_s_obj[:, 0], mean_yerr_OPT_s_obj[:, 1],
                yerr=mean_yerr_OPT_s_obj[:, 2], linestyle='None', color='green',capsize=2,elinewidth=.8)
    ax.plot(mean_yerr_OPT_s_obj[:, 0], mean_yerr_OPT_s_obj[:, 1], color='green', label=r"OPT$_{\rm S}$",
            linewidth=2)

    ax.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], yerr=mean_yerr_OPT_obj[:, 2], color='red',
                linestyle='None',capsize=2,elinewidth=.8)
    ax.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='red', label=r'OPT', linewidth=2)

    ax.grid(True)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([100, 1500])

def plot_subfig_ar(name, ax, dump_dir, ar_type='opt(s)'):

    x = np.arange(100, 1500 + 1, 100)
    x = x.reshape((len(x), 1))

    mean_yerr_greedy_00 = None; mean_yerr_greedy_25 = None;mean_yerr_greedy_50 = None;mean_yerr_greedy_75 = None;

    if ar_type =='opt':
        f = np.load(dump_dir + '/' + name[0] + ".npz")
        mean_yerr_greedy_00 = f["mean_yerr_greedy_ar"]
        f = np.load(dump_dir + '/' + name[1] + ".npz")
        mean_yerr_greedy_25 = f["mean_yerr_greedy_ar"]
        f = np.load(dump_dir + '/' + name[2] + ".npz")
        mean_yerr_greedy_50 = f["mean_yerr_greedy_ar"]
        f = np.load(dump_dir + '/' + name[3] + ".npz")
        mean_yerr_greedy_75 = f["mean_yerr_greedy_ar"]
        # x = np.arange(100, 1500 + 1, 100)
        # x = x.reshape((len(x), 1))
        # greedy_ar_75 = f["greedy_ar"]
        # mean_yerr_greedy_75 = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_ar_75)), 1)

    elif ar_type =='opt(s)':
        f = np.load(dump_dir + '/' + name[0] + ".npz")
        greedy_obj = f["greedy_obj"]
        OPT_s_obj = f["OPT_s_obj"]
        greedy_ar = greedy_obj/OPT_s_obj
        mean_yerr_greedy_00 =np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_ar)), 1)

        f = np.load(dump_dir + '/' + name[1] + ".npz")
        greedy_obj = f["greedy_obj"]
        OPT_s_obj = f["OPT_s_obj"]
        greedy_ar = greedy_obj/OPT_s_obj
        mean_yerr_greedy_25 =np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_ar)), 1)

        f = np.load(dump_dir + '/' + name[2] + ".npz")
        greedy_obj = f["greedy_obj"]
        OPT_s_obj = f["OPT_s_obj"]
        greedy_ar = greedy_obj/OPT_s_obj
        mean_yerr_greedy_50 =np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_ar)), 1)

        f = np.load(dump_dir + '/' + name[3] + ".npz")
        greedy_obj = f["greedy_obj"]
        OPT_s_obj = f["OPT_s_obj"]
        greedy_ar = greedy_obj/OPT_s_obj
        mean_yerr_greedy_75 =np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_ar)), 1)

    ax.errorbar(mean_yerr_greedy_00[:, 0], mean_yerr_greedy_00[:, 1],
                yerr=mean_yerr_greedy_00[:, 2], linestyle='None', color='blue')
    ax.plot(mean_yerr_greedy_00[:, 0], mean_yerr_greedy_00[:, 1], color='blue', marker='.', label="0%",
            linewidth=2,
            linestyle='-')

    ax.errorbar(mean_yerr_greedy_25[:, 0], mean_yerr_greedy_25[:, 1], yerr=mean_yerr_greedy_25[:, 2], color='red',
                linestyle='None')
    ax.plot(mean_yerr_greedy_25[:, 0], mean_yerr_greedy_25[:, 1], linestyle = '-', color='red', label=r'25%', linewidth=2)


    ax.errorbar(mean_yerr_greedy_50[:, 0], mean_yerr_greedy_50[:, 1],
                yerr=mean_yerr_greedy_50[:, 2], linestyle='None', color='green')
    ax.plot(mean_yerr_greedy_50[:, 0], mean_yerr_greedy_50[:, 1], color='green', label=r"50%",
            linewidth=2, linestyle='-')

    ax.errorbar(mean_yerr_greedy_75[:, 0], mean_yerr_greedy_75[:, 1],
                yerr=mean_yerr_greedy_75[:, 2], linestyle='None', color='black')
    ax.plot(mean_yerr_greedy_75[:, 0], mean_yerr_greedy_75[:, 1], color='black', label=r"75% (% elastic customers)  ",
            linewidth=2, linestyle='-')


    ar = map(lambda n:1./(np.floor(1/np.cos(72/180. *np.pi) * 1/np.cos(72/180./2. *np.pi) )+1) ,mean_yerr_greedy_75[:, 0])
    # ar = map(lambda n:1./(np.floor(( 1/np.cos(72/180. *np.pi)) * 1/np.cos(72/180./2. *np.pi) + 1)*(2*np.log2(n) + 1)) ,mean_yerr_greedy_75[:, 0])
    # ar = map(lambda n:1./(2*np.log2(n) + 1)  ,mean_yerr_greedy_75[:, 0])
    ax.plot(mean_yerr_greedy_75[:, 0], ar, color='black',
            linewidth=2, linestyle='-.')
    ax.text(568, .12, r'Theoretical', fontsize=13)

    print ar

    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([100, 1500])

def plot_subfig_time(name, ax, dump_dir):

    f = np.load(dump_dir + '/' + name + ".npz")
    mean_yerr_greedy_time = f['mean_yerr_greedy_time']
    # mean_yerr_greedy_time[:,1:] = 1000 * mean_yerr_greedy_time[:,1:]
    mean_yerr_OPT_time = f['mean_yerr_OPT_time']
    # mean_yerr_OPT_time[:,1:] = 1000 * mean_yerr_OPT_time[:,1:]
    mean_yerr_OPT_s_time = f['mean_yerr_OPT_s_time']

    ax.errorbar(mean_yerr_greedy_time[:, 0], mean_yerr_greedy_time[:, 1],
                yerr=mean_yerr_greedy_time[:, 2], linestyle='None', color='blue')
    ax.plot(mean_yerr_greedy_time[:, 0], mean_yerr_greedy_time[:, 1], color='blue', marker='.', label=r'GreedyDisDem',
            linewidth=2,
            linestyle='-.')

    # c = ax.twinx()

    ax.errorbar(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1], yerr=mean_yerr_OPT_time[:, 2], color='red',
                linestyle='None')
    ax.plot(mean_yerr_OPT_time[:, 0], mean_yerr_OPT_time[:, 1], color='red', label=r'Numerical (Gurobi)', linewidth=2)


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
def plot_subfig_loss(name, ax, dump_dir):

    f = np.load(dump_dir + '/' + name + ".npz")
    greedy_loss = 100*f["adaptive_greedy_loss_ratio"]
    OPT_s_loss= 100*f["adaptive_OPT_s_loss_ratio"]
    print  name
    print 'max greedy loss %.3f %%'% np.max(greedy_loss)
    print 'max OPT(s) loss %.3f %%'% np.max(OPT_s_loss)
    print '=========================='

    x = np.arange(100, 1500 + 1, 100)
    x = x.reshape((len(x), 1))
    mean_yerr_greedy_loss = np.append(x, np.array(map(lambda y: u.mean_yerr(y), greedy_loss)), 1)
    mean_yerr_OPT_s_loss = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_s_loss)), 1)

    ax.errorbar(mean_yerr_greedy_loss[:, 0], mean_yerr_greedy_loss[:, 1],
                yerr=mean_yerr_greedy_loss[:, 2], linestyle='None', color='blue',capsize=2,elinewidth=.8)
    ax.plot(mean_yerr_greedy_loss[:, 0], mean_yerr_greedy_loss[:, 1], color='blue', marker='.', label="GreedyOPF",
            linewidth=2,
            linestyle='-')


    ax.errorbar(mean_yerr_OPT_s_loss[:, 0], mean_yerr_OPT_s_loss[:, 1],
                yerr=mean_yerr_OPT_s_loss[:, 2], linestyle='None', color='green',capsize=2,elinewidth=.8)
    ax.plot(mean_yerr_OPT_s_loss[:, 0], mean_yerr_OPT_s_loss[:, 1], color='green', label=r"OPT$_{\rm S}$",
            linewidth=2, linestyle='-')
    ax.grid(True)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # format_exponent(ax, 'y')

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([100, 1500])
    ax.set_ylim([0, 4])
def plot_all_obj(dump_dir="results/dump/",network=123, fig_dir="results/"):
    """
    :param dump_dir:
    :param fig_dir:
    :return:
    """
    plt.ioff()
    # plt.clf()
    if network == 123:
        FCR_name = "sim-123-opts-little-bit-large/adapt__FCR__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FCM_name = "sim-123-opts-little-bit-large/adapt__FCM__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FUR_name = "sim-123-opts-little-bit-large/adapt__FUR__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FUM_name = "sim-123-opts-little-bit-large/adapt__FUM__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
    else:
        FCR_name = "adapt__FCR_F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40"
        FCM_name = "adapt__FCM_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=100"
        FUR_name = "adapt__FUR_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FUM_name = "adapt__FUM_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=100"

    # fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7.2, 4))

    plot_subfig_obj(name=FCR_name, ax=fcr, dump_dir=dump_dir)
    fcr.set_title('CR')

    fcr.legend(bbox_to_anchor=(0.4, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    plot_subfig_obj(name=FCM_name, ax=fcm, dump_dir=dump_dir)
    fcm.set_title('CM')
    # fcm.set_ylim([0,7*10**12])

    plot_subfig_obj(name=FUR_name, ax=fur, dump_dir=dump_dir)
    fur.set_title('UR')

    plot_subfig_obj(name=FUM_name, ax=fum, dump_dir=dump_dir)
    fum.set_title('UM')
    # fum.set_ylim([0,1.2*10**7])

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Objective value', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    if network == 123:
        plt.savefig(fig_dir + "FnT_TCNS_obj_123.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir + "FnT_TCNS_obj_38.pdf", bbox_inches='tight')


def plot_all_ar(dump_dir="results/dump/", network=123, fig_dir="results/", ar_type="opt"):
    """
    :param dump_dir:
    :param fig_dir:
    :param type: "opt"|"opt(s)"  (calculate approximation ratio vs opt or opt(s)
    :return:
    """
    plt.ioff()
    # plt.clf()

    if network == 123:
        FCR_name = ["adapt__FCR__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCR__topology=123__F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCR__topology=123__F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCR__topology=123__F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=40"]
        FCM_name = ["adapt__FCM__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCM__topology=123__F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCM__topology=123__F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCM__topology=123__F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=40"]
        FUR_name = ["adapt__FUR__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUR__topology=123__F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUR__topology=123__F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUR__topology=123__F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=40"]
        FUM_name = ["sim-123-opts-little-bit-large/adapt__FUM__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUM__topology=123__F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUM__topology=123__F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUM__topology=123__F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=40"]
    elif network==38:
        FCR_name = ["adapt__FCR_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCR_F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCR_F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FCR_F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=40"]
        FCM_name = ["adapt__FCM_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=100",
                    "adapt__FCM_F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=100",
                    "adapt__FCM_F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=100",
                    "adapt__FCM_F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=100"] ###### FIX THIS GUY ####
        FUR_name = ["adapt__FUR_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUR_F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUR_F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=40",
                    "adapt__FUR_F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=40"]
        FUM_name = ["adapt__FUM_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=100",
                    "adapt__FUM_F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=100",
                    "adapt__FUM_F_percentage=0.50_max_n=1500_step_n=100_start_n=100_reps=100",
                    "adapt__FUM_F_percentage=0.75_max_n=1500_step_n=100_start_n=100_reps=100"]




    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 4))

    plot_subfig_ar(name=FCR_name, ax=fcr, dump_dir=dump_dir, ar_type=ar_type)
    fcr.set_title('CR')

    fcr.legend(bbox_to_anchor=(0., 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    plot_subfig_ar(name=FCM_name, ax=fcm, dump_dir=dump_dir, ar_type=ar_type)
    fcm.set_title('CM')

    plot_subfig_ar(name=FUR_name, ax=fur, dump_dir=dump_dir, ar_type=ar_type)
    fur.set_title('UR')

    plot_subfig_ar(name=FUM_name, ax=fum, dump_dir=dump_dir, ar_type=ar_type)
    fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Approximation ratio', ha='center', va='center', rotation='vertical', fontsize=14)
    fcr.set_ylim([0, 1])
    fcm.set_ylim([0, 1])
    fur.set_ylim([0, 1])
    fum.set_ylim([0, 1])

    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    if network==123:
        plt.savefig(fig_dir +"TCNS_greedy_ar_123_"+ar_type+"_adapt.pdf", bbox_inches='tight')
    elif network==38:
        plt.savefig(fig_dir +"TCNS_greedy_ar_38_"+ar_type+"_adapt.pdf", bbox_inches='tight')

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
    # fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
    fcr.legend(bbox_to_anchor=(.40, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
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

    # plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
    plt.tight_layout(pad=1, w_pad=0.2, h_pad=0.2)
    plt.savefig(fig_dir + "TCNS_greedy_time_123_all.pdf", bbox_inches='tight')

def plot_all_loss(dump_dir="results/dump/", network=123, fig_dir="results/"):
    """
    :param dump_dir:
    :param fig_dir:
    :return:
    """
    plt.ioff()
    # plt.clf()
    if network == 123:
        FCR_name = "adapt__FCR__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FCM_name = "adapt__FCM__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FUR_name = "adapt__FUR__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FUM_name = "adapt__FUM__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
    else:
        FCR_name = "adapt__FCR_F_percentage=0.25_max_n=1500_step_n=100_start_n=100_reps=40"
        FCM_name = "adapt__FCM_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=100"
        FUR_name = "adapt__FUR_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40"
        FUM_name = "adapt__FUM_F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=100"



    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7.2,4))
    # fig, (fcm, fum) = plt.subplots(1, 2, figsize=(7, 3))

    plot_subfig_loss(name=FCR_name, ax=fcr, dump_dir=dump_dir)
    fcr.set_title('CR')
    fcr.legend(bbox_to_anchor=(0.6, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)



    plot_subfig_loss(name=FCM_name, ax=fcm, dump_dir=dump_dir)
    fcm.set_title('CM')
    # fcm.set_ylim([0,5*10**12])

    plot_subfig_loss(name=FUR_name, ax=fur, dump_dir=dump_dir)
    fur.set_title('UR')

    plot_subfig_loss(name=FUM_name, ax=fum, dump_dir=dump_dir)
    fum.set_title('UM')
    # fum.set_ylim([0,1.2*10**7])

    fig.text(0.5, 0.01, 'Number of customers', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, r'$\delta \times 100$', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    if network == 123:
        plt.savefig(fig_dir + "FnT_TCNS_loss_123.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir + "FnT_TCNS_loss_38.pdf", bbox_inches='tight')
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
    g.plot(mean_yerr_gr_time[:, 0], mean_yerr_gr_time[:, 1], color='blue', label="GreedysOPF", linewidth=2,
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
