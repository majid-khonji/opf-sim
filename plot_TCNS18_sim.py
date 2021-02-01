__author__ = 'Majid Khonji'
# revision 2: 1st review updated

import matplotlib.pyplot as plt
import numpy as np
import util as u

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


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


def _subfig_ar(name, ax, dump_dir, start_n=100, step_n=100, max_n=1500):
    f = np.load(dump_dir + '/' + name + ".npz")
    round_OPF_ar = f['round_OPF_ar']
    # OPT_ar = f['OPT_ar']
    # frac_OPF_ar = f['frac_OPF_ar']

    x = np.arange(start_n, max_n + 1, step_n)
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

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    _format_exponent(ax, 'y')
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([start_n, max_n])


# modified (takes the max for OPT)
def _subfig_obj(filename, ax, dump_dir, start_n=100, max_n=3500, step_n=100):
    f = np.load(dump_dir + '/' + filename + ".npz")

    print "plotting %s..." % filename
    round_OPF_obj = f['round_OPF_obj']
    OPT_obj = f['OPT_obj']
    # frac_OPF_obj = f['frac_OPF_obj']

    x = np.arange(start_n, max_n + 1, step_n)
    x = x.reshape((len(x), 1))

    print round_OPF_obj.shape
    print x.shape

    mean_yerr_round_OPF_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_obj)), 1)
    mean_yerr_frac_OPF_obj = f['mean_yerr_frac_OPF_obj']
    mean_yerr_OPT_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_obj)), 1)

    ax.errorbar(mean_yerr_round_OPF_obj[:, 0], mean_yerr_round_OPF_obj[:, 1],
                yerr=mean_yerr_round_OPF_obj[:, 2], elinewidth=.5, capsize=1.5, color='dodgerblue')
    ax.plot(mean_yerr_round_OPF_obj[:, 0], mean_yerr_round_OPF_obj[:, 1], color='dodgerblue', marker='.', label="PTAS",
            linewidth=2,
            linestyle='-.')

    ax.errorbar(mean_yerr_frac_OPF_obj[:, 0], mean_yerr_frac_OPF_obj[:, 1], yerr=mean_yerr_frac_OPF_obj[:, 2],
                elinewidth=.5, capsize=1.5, color='green')
    ax.plot(mean_yerr_frac_OPF_obj[:, 0], mean_yerr_frac_OPF_obj[:, 1], color='darkgreen', label=r'Frac. (Lower Bound)',
            linewidth=2)

    ax.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], yerr=mean_yerr_OPT_obj[:, 2], elinewidth=.5,
                capsize=1.5, color='darkred')
    ax.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='darkred', label=r'Numerical (Gurobi)', linewidth=2)

    ax.grid(True)

    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([start_n, max_n])
    plt.setp(ax, xticks=np.arange(start_n, max_n + 1, step_n * 4))


def _subfig_obj2(filename, ax, dump_dir='results/dump', y_lim=None, start_n=100, step_n=100, max_n=3500):
    x = range(start_n, max_n + 1, step_n)

    f = np.load(dump_dir + '/' + filename + ".npz")
    opt_data = f['OPT_obj'].tolist()

    f = np.load(dump_dir + '/' + filename + ".npz")
    alg_data = f['round_OPF_obj'].tolist()

    f = np.load(dump_dir + '/' + filename + ".npz")
    no_LP_alg_data = f['no_LP_round_OPF_obj'].tolist()

    __box_plot(ax, alg_data, name='Alg.', darkcolor='dodgerblue', lightcolor='lightblue')
    # __box_plot(ax, no_LP_alg_data, name='No LP', darkcolor='green', lightcolor='lightgreen')
    __box_plot(ax, opt_data, name='Gurobi OPT', darkcolor='darkred', lightcolor='indianred')

    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    _format_exponent(ax, 'y')
    plt.setp(ax, xticks=range(1, len(x) + 1, 4))
    ax.set_xticklabels(range(start_n, max_n + 1, 4 * step_n))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    # ax.set_xlim([start_n, max_n])
    if y_lim != None:
        ax.set_ylim(y_lim)


def plot_all_obj(alg='PTAS', dump_dir="results/dump/", network=123, fig_dir="results/"):
    plt.ioff()
    if network == 123:
        if alg == 'PTAS':
            FCR_name = 'TCNS18:[FCR]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUR_name = 'TCNS18:[FUR]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FCM_name = 'TCNS18:[FCM]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUM_name = 'TCNS18:[FUM]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        else:
            FCR_name = 'TCNS18:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FCM_name = 'TCNS18:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUR_name = 'TCNS18:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUM_name = 'TCNS18:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    elif network == 13:
        if alg == 'PTAS':
            FCR_name = 'TCNS18:[FCR]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUR_name = 'TCNS18:[FUR]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FCM_name = 'TCNS18:[FCM]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUM_name = 'TCNS18:[FUM]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        else:
            FCR_name = 'TCNS18:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FCM_name = 'TCNS18:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUR_name = 'TCNS18:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
            FUM_name = 'TCNS18:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    else:
        print "wrong network"
        return

    # fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 4.00))

    if network == 13:
        _subfig_obj(filename=FCR_name, ax=fcr, dump_dir=dump_dir, max_n=3500, start_n=100, step_n=100)
        fcr.legend(bbox_to_anchor=(.15, 1.20, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        _subfig_obj(filename=FCM_name, ax=fcm, dump_dir=dump_dir, max_n=3500, start_n=100, step_n=100)
        _subfig_obj(filename=FUR_name, ax=fur, dump_dir=dump_dir, max_n=3500, start_n=100, step_n=100)
        _subfig_obj(filename=FUM_name, ax=fum, dump_dir=dump_dir, max_n=3500, start_n=100, step_n=100)
    else:
        _subfig_obj(filename=FCR_name, ax=fcr, dump_dir=dump_dir)
        fcr.legend(bbox_to_anchor=(0.08, 1.20, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        _subfig_obj(filename=FCM_name, ax=fcm, dump_dir=dump_dir)
        _subfig_obj(filename=FUR_name, ax=fur, dump_dir=dump_dir)
        _subfig_obj(filename=FUM_name, ax=fum, dump_dir=dump_dir)

    fcr.set_title('CR')
    fcm.set_title('CM')
    fur.set_title('UR')
    fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Objective value', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=0.2, h_pad=.2)
    if network == 123:
        plt.savefig(fig_dir + "TCNS18_obj_123.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir + "TCNS18_obj_13.pdf", bbox_inches='tight')


def _subfig_ar_1x2(data, ax, dump_dir, start_n=100, step_n=100, max_n=3500):
    x = np.arange(start_n, max_n + 1, step_n)
    x = x.reshape((len(x), 1))

    # f = np.load(dump_dir + '/' + data['fcr']+ ".npz")
    # round_OPF_ar = f['round_OPF_ar']
    # print 'plotting FCR...'
    # print x.shape, round_OPF_ar.shape
    # mean_yerr_round_OPF_ar = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_ar)), 1)
    # ax.errorbar(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1],
    #             yerr=mean_yerr_round_OPF_ar[:, 2], linestyle='None', color='black')
    # ax.plot(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1], color='black', marker='.', label="CR",
    #         linewidth=2,
    #         linestyle='-.')

    # f = np.load(dump_dir + '/' + data['fcm']+ ".npz")
    # round_OPF_ar = f['round_OPF_ar2']
    # print 'plotting FCM...'
    # print x.shape, round_OPF_ar.shape
    # mean_yerr_round_OPF_ar = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_ar)), 1)
    #
    # ax.errorbar(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1],
    #             yerr=mean_yerr_round_OPF_ar[:, 2], linestyle='None', color='blue')
    # ax.plot(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1], color='blue', marker='.', label="CM",
    #         linewidth=2,
    #         linestyle='-.')

    f = np.load(dump_dir + '/' + data['fur'] + ".npz")
    round_OPF_ar = f['round_OPF_ar']
    print 'plotting FUR...'
    print x.shape, round_OPF_ar.shape
    mean_yerr_round_OPF_ar = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_ar)), 1)
    ax.errorbar(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1],
                yerr=mean_yerr_round_OPF_ar[:, 2], linestyle='None', color='maroon')
    ax.plot(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1], color='maroon', marker='.', label="UR",
            linewidth=2,
            linestyle='-.')

    f = np.load(dump_dir + '/' + data['fum'] + ".npz")
    round_OPF_ar = f['round_OPF_ar']
    print 'plotting FUM...'
    print x.shape, round_OPF_ar.shape
    mean_yerr_round_OPF_ar = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_ar)), 1)
    ax.errorbar(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1],
                yerr=mean_yerr_round_OPF_ar[:, 2], linestyle='None', color='green')
    ax.plot(mean_yerr_round_OPF_ar[:, 0], mean_yerr_round_OPF_ar[:, 1], color='green', marker='.', label="UM",
            linewidth=2,
            linestyle='-.')

    ax.grid(True)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    _format_exponent(ax, 'y')
    plt.setp(ax, xticks=np.arange(start_n, max_n + 1, step_n * 4))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([start_n + 100, max_n])
    ax.set_ylim([.8, 1.3])


def plot_ar_2x1(dump_dir="results/dump/", fig_dir="results/"):
    plt.ioff()
    plt.clf()
    fig, (net_13, net_123) = plt.subplots(1, 2, sharex='col', figsize=(7, 2.5))

    FCR_name = 'TCNS18:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TCNS18:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TCNS18:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TCNS18:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_ar_1x2(data, ax=net_13, dump_dir=dump_dir)
    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.7, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    FCR_name = 'TCNS18:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TCNS18:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TCNS18:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TCNS18:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_ar_1x2(data, ax=net_123, dump_dir=dump_dir)
    net_123.set_title('IEEE 123-node network')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, 'Approximation ratio', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=-.8, h_pad=0.2)
    plt.savefig(fig_dir + "TCNS18_ar.pdf", bbox_inches='tight')


def plot_ar_4x4(dump_dir="results/dump/", network=13, fig_dir="results/", ar_type="opt"):
    plt.ioff()
    plt.clf()
    if network == 123:
        FCR_name = 'TCNS18:[FCR]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUR_name = 'TCNS18:[FUR]__topology=123__F_percentage=0.00_max_n=1500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
    elif network == 13:
        FCR_name = 'TCNS18:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FCM_name = 'TCNS18:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUR_name = 'TCNS18:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
        FUM_name = 'TCNS18:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=2000_reps=40'
    else:
        print "wrong network"
        return

    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 5))

    if network == 13:
        _subfig_ar(name=FCR_name, ax=fcr, dump_dir=dump_dir, max_n=3500, start_n=2000, step_n=100)
        fcr.set_title('CR')
        fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        _subfig_ar(name=FCM_name, ax=fcm, dump_dir=dump_dir, max_n=3500, start_n=2000, step_n=100)
        fcm.set_title('CM')
        _subfig_ar(name=FUR_name, ax=fur, dump_dir=dump_dir, max_n=3500, start_n=2000, step_n=100)
        fur.set_title('UR')
        _subfig_ar(name=FUM_name, ax=fum, dump_dir=dump_dir, max_n=3500, start_n=2000, step_n=100)
        fum.set_title('UM')
    else:
        _subfig_ar(name=FCR_name, ax=fcr, dump_dir=dump_dir)
        fcr.set_title('CR')
        fcr.legend(bbox_to_anchor=(0., 1.15, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        _subfig_ar(name=FCM_name, ax=fcm, dump_dir=dump_dir, max_n=3500, start_n=2000, step_n=100)
        fcm.set_title('CM')
        _subfig_ar(name=FUR_name, ax=fur, dump_dir=dump_dir)
        fur.set_title('UR')
        _subfig_ar(name=FUM_name, ax=fum, dump_dir=dump_dir, max_n=3500, start_n=2000, step_n=100)
        fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, 'Approximation ratio', ha='center', va='center', rotation='vertical', fontsize=14)
    if network == 13:
        fcr.set_ylim([.0, 10.5])
        fcm.set_ylim([.0, 10.5])
        fur.set_ylim([.0, 10.5])
        fum.set_ylim([.0, 10.5])
        plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
        plt.savefig(fig_dir + "TCNS18_ar_13.pdf", bbox_inches='tight')
        print 'saving'
    elif network == 123:
        fcr.set_ylim([.9, 1.2])
        fcm.set_ylim([.9, 1.2])
        fur.set_ylim([.9, 1.2])
        fum.set_ylim([.9, 1.2])
        plt.tight_layout(pad=1, w_pad=.8, h_pad=0.2)
        plt.savefig(fig_dir + "TCNS18_ar_123.pdf", bbox_inches='tight')


def __box_plot(ax, data,x=None,name='CM', darkcolor='dodgerblue', lightcolor='lightblue'):
    medians = [np.median(d) for d in data]
    medians = [medians[0]] + medians
    boxprops = dict(linestyle='-', linewidth=1, color=darkcolor, facecolor=lightcolor)
    flierprops = dict(marker='+', alpha=.4, markeredgecolor=darkcolor, markersize=5)
    medianprops = dict(linestyle='-', color=darkcolor, linewidth=1)
    wiskerprops = dict(linestyle='-', color=darkcolor, linewidth=1)
    capprops = dict(linestyle='-', color=darkcolor, linewidth=1)
    ax.plot(medians, color=darkcolor, alpha=.6, label=name)
    r = ax.boxplot(data,positions=x, patch_artist=True, capprops=capprops, whiskerprops=wiskerprops, boxprops=boxprops,
                   medianprops=medianprops, flierprops=flierprops)
    print "__box_plot called.."
    count_outliars = []
    for i in range(len(data)):
        top_points = r["fliers"][i].get_data()[1]
        count_outliars.append(len(top_points))
    print count_outliars


def _subfig_net_gen(data, ax, num_edges=1, percentage=True, field='round_OPF_frac_count', dump_dir='results/dump',
                    y_lim=None, start_n=100, step_n=100, max_n=3500):
    if step_n < 1:
        x = np.arange(start_n, max_n+start_n, step_n)
    else:
        x = range(start_n, max_n + 1, step_n)

    # for ones
    if num_edges == None:
        f = np.load(dump_dir + '/' + data['fcr'] + ".npz")
        fcr_data = (f[field] / f["OPT_ones_count"]).tolist()

        f = np.load(dump_dir + '/' + data['fcm'] + ".npz")
        fcm_data = (f[field] / f["OPT_ones_count"]).tolist()

        f = np.load(dump_dir + '/' + data['fur'] + ".npz")
        fur_data = (f[field] / f["OPT_ones_count"]).tolist()

        f = np.load(dump_dir + '/' + data['fum'] + ".npz")
        fum_data = (f[field] / f["OPT_ones_count"]).tolist()

    else:
        f = np.load(dump_dir + '/' + data['fcr'] + ".npz")
        fcr_data = (f[field] / float(num_edges)).tolist()

        f = np.load(dump_dir + '/' + data['fcm'] + ".npz")
        fcm_data = (f[field] / float(num_edges)).tolist()

        f = np.load(dump_dir + '/' + data['fur'] + ".npz")
        fur_data = (f[field] / float(num_edges)).tolist()

        f = np.load(dump_dir + '/' + data['fum'] + ".npz")
        fum_data = (f[field] / float(num_edges)).tolist()

    __box_plot(ax, fcr_data, name='CR', darkcolor='darkgray', lightcolor='lightgray')
    __box_plot(ax, fcm_data, name='CM', darkcolor='dodgerblue', lightcolor='lightblue')
    __box_plot(ax, fur_data, name='UR', darkcolor='darkgreen', lightcolor='lightgreen')
    __box_plot(ax, fum_data, name='UM', darkcolor='darkred', lightcolor='indianred')

    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # _format_exponent(ax, 'y')
    # manipulate
    # vals = ax.get_yticks()
    # ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
    if percentage:
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    if step_n >= 1:
        plt.setp(ax, xticks=range(1, len(x) + 1, 4))
        ax.set_xticklabels(range(start_n, max_n + 1, 4 * step_n))
    else:
        ticks = np.arange(step_n, max_n+step_n,step_n)
        formated_ticks = ['%1.1f'%i for i in ticks]
        plt.setp(ax, xticks=range(1,len(ticks)+1))
        ax.set_xticklabels(formated_ticks)


    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # ax.set_xlim([start_n, max_n])
    if y_lim != None:
        ax.set_ylim(y_lim)


def plot_frac_over_ones(dump_dir="results/dump/", y_label='Percentage of fractional\n components over Ones',
                        fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (net_13, net_123) = plt.subplots(1, 2, sharex='col', figsize=(7.2, 2.5))

    FCR_name = 'TCNS18:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TCNS18:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TCNS18:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TCNS18:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_13, num_edges=None, field='round_OPF_frac_count', dump_dir=dump_dir)
    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.30, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    FCR_name = 'TCNS18:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TCNS18:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TCNS18:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TCNS18:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_123, num_edges=None, field='round_OPF_frac_count', dump_dir=dump_dir)
    net_123.set_title('IEEE 123-node network')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(-0.01, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=0.2, h_pad=0.2)
    plt.savefig(fig_dir + "TCNS18_frac_over_ones.pdf", bbox_inches='tight')


def plot_frac_comp_count(alg='PTAS', dump_dir="results/dump/",
                         y_label='Percentage of fractional\n components over $4m$', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (net_13, net_123) = plt.subplots(1, 2, sharex='col', figsize=(7.2, 2.5))

    if alg == 'PTAS':
        FCR_name = 'TCNS18:[FCR]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    else:
        FCR_name = 'TCNS18:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_13, num_edges=4 * 13., field='round_OPF_frac_count', dump_dir=dump_dir)
    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.30, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    if alg == 'PTAS':
        FCR_name = 'TCNS18:[FCR]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    else:
        FCR_name = 'TCNS18:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_123, num_edges=4 * 123., field='round_OPF_frac_count', dump_dir=dump_dir)
    net_123.set_title('IEEE 123-node network')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(-0.01, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=0.2, h_pad=0.2)
    plt.savefig(fig_dir + "TCNS18_frac_count.pdf", bbox_inches='tight')


def plot_ar(alg='PTAS', dump_dir="results/dump/", y_label='Approximation Ratio', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (net_13, net_123) = plt.subplots(1, 2, sharex='col', figsize=(7, 2.5))
    if alg == 'PTAS':
        FCR_name = 'TCNS18:[FCR]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__max_resample=10__epsilon=0.10__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    else:
        FCR_name = 'TCNS18:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_13, percentage=False, field='round_OPF_ar', y_lim=[.96, 1.8], dump_dir=dump_dir)
    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.30, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    if alg == 'PTAS':
        FCR_name = 'TCNS18:[FCR]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__max_resample=10__epsilon=0.10__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    else:
        FCR_name = 'TCNS18:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TCNS18:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TCNS18:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TCNS18:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_123, percentage=False, field='round_OPF_ar', y_lim=[.96, 1.8], dump_dir=dump_dir)
    net_123.set_title('IEEE 123-node network')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir + "TCNS18_ar.pdf", bbox_inches='tight')


def plot_e_ar(dump_dir="results/dump/", y_label='Approximation Ratio', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (net_13, net_123) = plt.subplots(1, 2, sharex='col', figsize=(7, 2.5))
    FCR_name = 'TCNS18_PTAS:[FCR]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
    FUR_name = 'TCNS18_PTAS:[FUR]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
    FCM_name = 'TCNS18_PTAS:[FCM]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
    FUM_name = 'TCNS18_PTAS:[FUM]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'

    FCR_name = '/TCNS18_PTAS_fixed/fcr_13'
    FUR_name = '/TCNS18_PTAS_fixed/fur_13'
    FCM_name = '/TCNS18_PTAS_fixed/fcm_13'
    FUM_name = '/TCNS18_PTAS_fixed/fum_13'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_13, percentage=False, field='round_OPF_ar', y_lim=[.9,2], dump_dir=dump_dir, start_n=.1, step_n=.1, max_n=.9)
    net_13.plot([1,9],[1.1,1.9], color='black', linestyle=':', linewidth=1)

    net_123.set_title('IEEE 123-node network')
    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.30, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)


    FCR_name = 'TCNS18_PTAS:[FCR]__topology=123__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
    FUR_name = 'TCNS18_PTAS:[FUR]__topology=123__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
    FCM_name = 'TCNS18_PTAS:[FCM]__topology=123__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
    FUM_name = 'TCNS18_PTAS:[FUM]__topology=123__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'

    FCR_name = '/TCNS18_PTAS_fixed/fcr_123_time'
    FUR_name = '/TCNS18_PTAS_fixed/fur_123_time'
    FCM_name = '/TCNS18_PTAS_fixed/fcm_123_time'
    FUM_name = '/TCNS18_PTAS_fixed/fum_123_time'

    data = {'fcm': FCM_name, 'fum': FUM_name, 'fcr': FCR_name, 'fur': FUR_name}
    _subfig_net_gen(data, ax=net_123, percentage=False, field='round_OPF_ar', y_lim=[.9,2], dump_dir=dump_dir, start_n=.1, step_n=.1, max_n=.9)
    net_123.plot([1,9],[1.1,1.9], color='black', linestyle=':', linewidth=1)

    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.30, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    fig.text(0.5, 0.01, '$\epsilon$', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir + "TCNS18_e_ar.pdf", bbox_inches='tight')


def _subfig_time(filename, ax, dump_dir='results/dump', y_lim=None, start_n=100, step_n=100, max_n=3500):
    if step_n < 1:
        x = np.arange(start_n, max_n+start_n, step_n)
        xp= None

    else:
        x = range(start_n, max_n + 1, step_n)
        xp = None

    f = np.load(dump_dir + '/' + filename + ".npz")
    opt_data = f['OPT_time'].tolist()

    f = np.load(dump_dir + '/' + filename + ".npz")
    alg_data = f['round_OPF_time'].tolist()

    # f = np.load(dump_dir + '/' + filename + ".npz")
    # no_LP_alg_data = f['no_LP_round_OPF_time'].tolist()

    __box_plot(ax, alg_data,x=xp, name='PTAS', darkcolor='dodgerblue', lightcolor='lightblue')
    # __box_plot(ax, no_LP_alg_data, name='No LP', darkcolor='green', lightcolor='lightgreen')
    __box_plot(ax, opt_data,x=xp, name='Numerical (Gurobi)', darkcolor='darkred', lightcolor='indianred')

    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    _format_exponent(ax, 'y')
    if step_n >= 1:
        plt.setp(ax, xticks=range(1, len(x) + 1, 4))
        ax.set_xticklabels(range(start_n, max_n + 1, 4 * step_n))
    else:
        # plt.setp(ax, xticks=.1*np.arange(1, len(x) + 1, 4))
        # ax.set_xticklabels(.1*np.arange(start_n, max_n + 1, 4 * step_n))
        ticks = np.arange(step_n, max_n+step_n,step_n)
        formated_ticks = ['%1.1f'%i for i in ticks]
        plt.setp(ax, xticks=range(1,len(ticks)+1))
        ax.set_xticklabels(formated_ticks)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # ax.set_xlim([start_n, max_n])
    if y_lim != None:
        ax.set_ylim(y_lim)


def plot_time(alg='PTAS', dump_dir="results/dump/", y_label='Running time (sec.)', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    FCR_name = 'TCNS18:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TCNS18:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TCNS18:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TCNS18:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    if alg=='PTAS':
        FCR_name = 'TCNS18_PTAS:[FCR]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
        FUR_name = 'TCNS18_PTAS:[FUR]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
        FCM_name = 'TCNS18_PTAS:[FCM]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
        FUM_name = 'TCNS18_PTAS:[FUM]__topology=13__max_resample=100__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'

        FCR_name = 'TCNS18_PTAS:[FCR]__topology=123__max_resample=99__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
        FUR_name = 'TCNS18_PTAS:[FUR]__topology=123__max_resample=99__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
        FCM_name = 'TCNS18_PTAS:[FCM]__topology=123__max_resample=99__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'
        FUM_name = 'TCNS18_PTAS:[FUM]__topology=123__max_resample=99__n=2500__F_percentage=0.00_max_e=0.90_step_e=0.10_start_e=0.10_reps=40'

        FCR_name = '/TCNS18_PTAS_fixed/fcr_123_time'
        FUR_name = '/TCNS18_PTAS_fixed/fur_123_time'
        FCM_name = '/TCNS18_PTAS_fixed/fcm_123_time'
        FUM_name = '/TCNS18_PTAS_fixed/fum_123_time'


    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 4.4))

    _subfig_time(FCR_name, ax=fcr, dump_dir=dump_dir, start_n=.1, step_n=.1,max_n=.9)
    fcr.legend(bbox_to_anchor=(.50, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
    _subfig_time(FCM_name, ax=fcm, dump_dir=dump_dir,y_lim=[0,120], start_n=.1, step_n=.1,max_n=.9)
    _subfig_time(FUR_name, ax=fur, dump_dir=dump_dir,  start_n=.1, step_n=.1,max_n=.9)
    _subfig_time(FUM_name, ax=fum, dump_dir=dump_dir,y_lim=[0,120], start_n=.1, step_n=.1,max_n=.9)

    fcr.set_title('CR')
    fcm.set_title('CM')
    fur.set_title('UR')
    fum.set_title('UM')

    fig.text(0.5, 0.01, '$\epsilon$', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=0, h_pad=0.2)
    if alg=='PTAS':
        plt.savefig(fig_dir + "TCNS18_PTAS_time.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir + "TCNS18_time.pdf", bbox_inches='tight')


if __name__ == "__main__":
    pass
