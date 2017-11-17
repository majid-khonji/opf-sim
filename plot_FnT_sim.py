__author__ = 'Majid Khonji'
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




# modified (takes the max for OPT)
def _subfig_obj(filename, ax, dump_dir, start_n=100, max_n=3500, step_n=100, alg1_name='PTAS',alg2_name=r'Frac. (Upper Bound)'):

    f = np.load(dump_dir + '/' + filename + ".npz")

    print "plotting %s..." % filename
    round_OPF_obj = f['round_OPF_obj']
    OPT_obj = f['OPT_obj']
    # frac_OPF_obj = f['frac_OPF_obj']

    x = np.arange(start_n, max_n +1 , step_n)
    x = x.reshape((len(x), 1))

    print round_OPF_obj.shape
    print x.shape

    mean_yerr_round_OPF_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), round_OPF_obj)), 1)
    mean_yerr_frac_OPF_obj = f['mean_yerr_frac_OPF_obj']
    mean_yerr_OPT_obj = np.append(x, np.array(map(lambda y: u.mean_yerr(y), OPT_obj)), 1)

    ax.errorbar(mean_yerr_round_OPF_obj[:, 0], mean_yerr_round_OPF_obj[:, 1],
                yerr=mean_yerr_round_OPF_obj[:, 2],elinewidth=.5,capsize=1.5, color='dodgerblue')
    ax.plot(mean_yerr_round_OPF_obj[:, 0], mean_yerr_round_OPF_obj[:, 1], color='dodgerblue', marker='.', label=alg1_name,
            linewidth=2,
            linestyle='-.')

    ax.errorbar(mean_yerr_frac_OPF_obj[:, 0], mean_yerr_frac_OPF_obj[:, 1], yerr=mean_yerr_frac_OPF_obj[:, 2],elinewidth=.5,capsize=1.5, color='green')
    ax.plot(mean_yerr_frac_OPF_obj[:, 0], mean_yerr_frac_OPF_obj[:, 1], color='darkgreen', label=alg2_name, linewidth=2)

    ax.errorbar(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], yerr=mean_yerr_OPT_obj[:, 2],elinewidth=.5,capsize=1.5, color='darkred')
    ax.plot(mean_yerr_OPT_obj[:, 0], mean_yerr_OPT_obj[:, 1], color='darkred', label=r'Numerical (Gurobi)', linewidth=2)




    ax.grid(True)

    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_xlim([start_n, max_n])
    plt.setp(ax, xticks=np.arange(start_n, max_n + 1,step_n*4))

def plot_csp_all_obj(dump_dir="results/dump/", fig_dir="results/"):
    plt.ioff()
    FCR_name = 'FnT:[FCR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'FnT:[FCM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'FnT:[FUR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'FnT:[FUM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    # fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7.2, 2.50))



    _subfig_obj(filename=FCR_name, ax=fcr, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100,
                alg1_name='CSP-PTAS', alg2_name=r'Frac. (Upper Bound)')
    fcr.legend(bbox_to_anchor=(.01, 1.20, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
    _subfig_obj(filename=FCM_name, ax=fcm, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100,
                alg1_name='CSP-PTAS', alg2_name=r'Frac. (Upper Bound)')
    _subfig_obj(filename=FUR_name, ax=fur, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100,
                alg1_name='CSP-PTAS', alg2_name=r'Frac. (Upper Bound)')
    _subfig_obj(filename=FUM_name, ax=fum, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100, alg1_name='CSP-PTAS', alg2_name=r'Frac. (Upper Bound)')

    fcr.set_title('CR')
    fcm.set_title('CM')
    fur.set_title('UR')
    fum.set_title('UM')


    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Objective value', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=0.2, h_pad=.2)
    plt.savefig(fig_dir + "FnT_sch_obj.pdf", bbox_inches='tight')

def plot_opf_all_obj(dump_dir="results/dump/", network=123, fig_dir="results/"):
    plt.ioff()
    if network == 123:
        FCR_name = 'TPS:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TPS:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TPS:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TPS:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    elif network==13:
        FCR_name = 'TPS:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FCM_name = 'TPS:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUR_name = 'TPS:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
        FUM_name = 'TPS:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    else:
        print "wrong network"
        return


    # fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 4.00))



    if network ==13:
        _subfig_obj(filename=FCR_name, ax=fcr, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')
        fcr.legend(bbox_to_anchor=(0, 1.20, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        _subfig_obj(filename=FCM_name, ax=fcm, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')
        _subfig_obj(filename=FUR_name, ax=fur, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')
        _subfig_obj(filename=FUM_name, ax=fum, dump_dir=dump_dir, max_n = 3500, start_n=100, step_n=100, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')
    else:
        _subfig_obj(filename=FCR_name, ax=fcr, dump_dir=dump_dir, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')
        fcr.legend(bbox_to_anchor=(0, 1.20, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
        _subfig_obj(filename=FCM_name, ax=fcm, dump_dir=dump_dir, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')
        _subfig_obj(filename=FUR_name, ax=fur, dump_dir=dump_dir, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')
        _subfig_obj(filename=FUM_name, ax=fum, dump_dir=dump_dir, alg1_name=r'PTAS-cOPF', alg2_name=r'Frac. (Lower Bound)')

    fcr.set_title('CR')
    fcm.set_title('CM')
    fur.set_title('UR')
    fum.set_title('UM')


    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Objective value', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=0.2, h_pad=.2)
    if network == 123:
        plt.savefig(fig_dir + "TPS_opf_obj_123.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir + "TPS_opf_obj_13.pdf", bbox_inches='tight')


def __box_plot(ax, data,name='CM', darkcolor='dodgerblue', lightcolor='lightblue'):
    medians = [np.median(d) for d in data]
    medians = [medians[0]] + medians
    boxprops = dict(linestyle='-', linewidth=1, color=darkcolor,facecolor=lightcolor)
    flierprops = dict(marker='+',alpha=.4,  markeredgecolor=darkcolor, markersize=5)
    medianprops = dict(linestyle='-', color=darkcolor, linewidth=1)
    wiskerprops = dict(linestyle='-', color=darkcolor, linewidth=1)
    capprops = dict(linestyle='-', color=darkcolor, linewidth=1)
    ax.plot(medians, color=darkcolor,alpha=.6, label=name)
    ax.boxplot(data, patch_artist=True, capprops=capprops, whiskerprops=wiskerprops,boxprops=boxprops, medianprops=medianprops, flierprops=flierprops)

def _subfig_net_gen(data, ax,num_edges=1,percentage=True, field='round_OPF_frac_comp_count', dump_dir='results/dump', y_lim=None, start_n=100, step_n=100, max_n=3500):

    x = range(start_n, max_n + 1 , step_n)

    f = np.load(dump_dir + '/' + data['fcr']+ ".npz")


    fcr_data = (f[field]/float(num_edges)).tolist()

    f = np.load(dump_dir + '/' + data['fcm']+ ".npz")
    fcm_data = (f[field]/float(num_edges)).tolist()

    f = np.load(dump_dir + '/' + data['fur']+ ".npz")
    fur_data = (f[field]/float(num_edges)).tolist()

    f = np.load(dump_dir + '/' + data['fum']+ ".npz")
    fum_data = (f[field]/float(num_edges)).tolist()

    __box_plot(ax, fcr_data, name='CR', darkcolor='darkgray', lightcolor='lightgray')
    __box_plot(ax, fcm_data, name='CM', darkcolor='dodgerblue', lightcolor='lightblue')
    __box_plot(ax, fur_data, name='UR',darkcolor='darkgreen',lightcolor='lightgreen')
    __box_plot(ax, fum_data, name='UM',darkcolor='darkred',lightcolor='indianred')




    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # _format_exponent(ax, 'y')
    # manipulate
    # vals = ax.get_yticks()
    # ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
    if percentage:
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.setp(ax, xticks=range(1,len(x)+1, 4))
    ax.set_xticklabels(range(start_n,max_n+1,4*step_n))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # ax.set_xlim([start_n, max_n])
    if y_lim != None:
        ax.set_ylim(y_lim)

def plot_csp_frac_comp_count(dump_dir="results/dump/", fig_dir="results/"):
    plt.ioff()

    fig, (frc ) = plt.subplots(1, 1, sharex='col', figsize=(4, 2.5))

    FCR_name = 'FnT:[FCR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'FnT:[FCM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'FnT:[FUR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'FnT:[FUM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum':FUM_name, 'fcr':FCR_name, 'fur':FUR_name}
    _subfig_net_gen(data,ax=frc,num_edges=2*24., field='round_OPF_frac_count',  dump_dir=dump_dir)
    frc.set_title('CSP-PTAS')
    frc.legend(bbox_to_anchor=(-.15, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    y_label='Percentage of fractional\n components over 48'
    fig.text(-0.02, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=0.2, h_pad=0.2)
    plt.savefig(fig_dir +"FnT_sch_frac_count.pdf", bbox_inches='tight')

def plot_opf_frac_comp_count(dump_dir="results/dump/", y_label='Percentage of fractional\n components over $4m$',  fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (net_13, net_123) = plt.subplots(1, 2, sharex='col', figsize=(7.2, 2.5))

    FCR_name = 'TPS:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TPS:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TPS:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TPS:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum':FUM_name, 'fcr':FCR_name, 'fur':FUR_name}
    _subfig_net_gen(data,ax=net_13,num_edges=4*13., field='round_OPF_frac_count',  dump_dir=dump_dir)
    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.30, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    FCR_name = 'TPS:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TPS:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TPS:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TPS:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum':FUM_name, 'fcr':FCR_name, 'fur':FUR_name}
    _subfig_net_gen(data, ax=net_123,num_edges=4*123., field='round_OPF_frac_count', dump_dir=dump_dir)
    net_123.set_title('IEEE 123-node network')


    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(-0.01, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=0.2, h_pad=0.2)
    plt.savefig(fig_dir +"TPS_opf_frac_count.pdf", bbox_inches='tight')
def plot_csp_ar(dump_dir="results/dump/", y_label='Approximation Ratio', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (ar ) = plt.subplots(1, 1, sharex='col', figsize=(4, 2.5))

    FCR_name = 'FnT:[FCR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'FnT:[FCM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'FnT:[FUR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'FnT:[FUM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum':FUM_name, 'fcr':FCR_name, 'fur':FUR_name}
    _subfig_net_gen(data,ax=ar,percentage=False,field='round_OPF_ar', y_lim=[0,1.05],  dump_dir=dump_dir)
    ar.set_title('CSP-PTAS')
    ar.legend(bbox_to_anchor=(-.15, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(-0.02, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir +"FnT_sch_ar.pdf", bbox_inches='tight')

def plot_opf_ar(dump_dir="results/dump/", y_label='Approximation Ratio', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (net_13, net_123) = plt.subplots(1, 2, sharex='col', figsize=(7, 2.5))

    FCR_name = 'TPS:[FCR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TPS:[FCM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TPS:[FUR]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TPS:[FUM]__topology=13__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum':FUM_name, 'fcr':FCR_name, 'fur':FUR_name}
    _subfig_net_gen(data,ax=net_13,percentage=False,field='round_OPF_ar', y_lim=[.96,1.8],  dump_dir=dump_dir)
    net_13.set_title('RBTS 13-node network')
    net_13.legend(bbox_to_anchor=(.30, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    FCR_name = 'TPS:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TPS:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TPS:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TPS:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'

    data = {'fcm': FCM_name, 'fum':FUM_name, 'fcr':FCR_name, 'fur':FUR_name}
    _subfig_net_gen(data, ax=net_123,percentage=False, field='round_OPF_ar',y_lim=[.96,1.8], dump_dir=dump_dir)
    net_123.set_title('IEEE 123-node network')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir +"TPS_opf_ar.pdf", bbox_inches='tight')
def _subfig_time(filename, ax, dump_dir='results/dump',alg_name='PTAS', y_lim=None, start_n=100, step_n=100, max_n=3500):

    x = range(start_n, max_n + 1 , step_n)

    f = np.load(dump_dir + '/' + filename+ ".npz")
    opt_data = f['OPT_time'].tolist()

    f = np.load(dump_dir + '/' + filename+ ".npz")
    alg_data = f['round_OPF_time'].tolist()

    f = np.load(dump_dir + '/' + filename+ ".npz")
    no_LP_alg_data = f['no_LP_round_OPF_time'].tolist()

    __box_plot(ax, alg_data, name=alg_name, darkcolor='dodgerblue', lightcolor='lightblue')
    # __box_plot(ax, no_LP_alg_data, name='No LP', darkcolor='green', lightcolor='lightgreen')
    __box_plot(ax, opt_data, name='Numerical (Gurobi)',darkcolor='darkred',lightcolor='indianred')

    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')
    plt.setp(ax, xticks=range(1,len(x)+1, 4))
    ax.set_xticklabels(range(start_n,max_n+1,4*step_n))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # ax.set_xlim([start_n, max_n])
    if y_lim != None:
        ax.set_ylim(y_lim)
def plot_csp_time(dump_dir="results/dump/", y_label='Running time (sec.)', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    FCR_name = 'FnT:[FCR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'FnT:[FCM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'FnT:[FUR]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'FnT:[FUM]__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    start_n = 100; max_n = 3500; step_n = 100
    x = range(start_n, max_n + 1 , step_n)

    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 4.4))

    _subfig_time(FCR_name, ax=fcr, dump_dir=dump_dir, alg_name='CSP-PTAS')
    fcr.legend(bbox_to_anchor=(.40, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
    _subfig_time(FCM_name, ax=fcm, dump_dir=dump_dir)
    _subfig_time(FUR_name, ax=fur, dump_dir=dump_dir)
    _subfig_time(FUM_name, ax=fum, dump_dir=dump_dir)

    fcr.set_title('CR')
    fcm.set_title('CM')
    fur.set_title('UR')
    fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=0, h_pad=0.2)
    plt.savefig(fig_dir +"FnT_sch_time.pdf", bbox_inches='tight')

def plot_opf_time(dump_dir="results/dump/", y_label='Running time (sec.)', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    FCR_name = 'TPS:[FCR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FCM_name = 'TPS:[FCM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUR_name = 'TPS:[FUR]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    FUM_name = 'TPS:[FUM]__topology=123__F_percentage=0.00_max_n=3500_step_n=100_start_n=100_reps=40'
    start_n = 100; max_n = 3500; step_n = 100
    x = range(start_n, max_n + 1 , step_n)

    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 4.4))

    _subfig_time(FCR_name, ax=fcr, dump_dir=dump_dir,alg_name='PTAS-cOPF')
    fcr.legend(bbox_to_anchor=(.4, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
    _subfig_time(FCM_name, ax=fcm, dump_dir=dump_dir)
    _subfig_time(FUR_name, ax=fur, dump_dir=dump_dir)
    _subfig_time(FUM_name, ax=fum, dump_dir=dump_dir)

    fcr.set_title('CR')
    fcm.set_title('CM')
    fur.set_title('UR')
    fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=0, h_pad=0.2)
    plt.savefig(fig_dir +"TPS_opf_time.pdf", bbox_inches='tight')

###################################################################################
##################################################################################3
########################### from TCNS #############################################
def _subfig_time_TCNS(filename, ax, dump_dir='results/dump',alg_name='PTAS', y_lim=None, start_n=100, step_n=100, max_n=2100):

    x = range(start_n, max_n + 1 , step_n)

    f = np.load(dump_dir + '/' + filename+ ".npz")
    opt_data = f['OPT_time'].tolist()

    f = np.load(dump_dir + '/' + filename+ ".npz")
    alg_data = f['greedy_time'].tolist()


    __box_plot(ax, alg_data, name=alg_name, darkcolor='dodgerblue', lightcolor='lightblue')
    # __box_plot(ax, no_LP_alg_data, name='No LP', darkcolor='green', lightcolor='lightgreen')
    __box_plot(ax, opt_data, name='Numerical (Gurobi)',darkcolor='darkred',lightcolor='indianred')

    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')
    plt.setp(ax, xticks=range(1,len(x)+1, 4))
    ax.set_xticklabels(range(start_n,max_n+1,4*step_n))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # ax.set_xlim([start_n, max_n])
    if y_lim != None:
        ax.set_ylim(y_lim)
def plot_sopf_time(dump_dir="results/dump/",network=123,  y_label='Running time (sec.)', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    if network == 123:
        FCR_name = "FCR__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
        FCM_name = "FCM__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
        FUR_name = "FUR__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
        FUM_name = "FUM__topology=123__F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
    else:
        FCR_name = "FCR_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
        FCM_name = "FCM_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
        FUR_name = "FUR_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"
        FUM_name = "FUM_F_percentage=0.00_max_n=2000_step_n=100_start_n=100_reps=100"


    fig, ((fcr, fcm), (fur, fum)) = plt.subplots(2, 2, sharex='col', figsize=(7, 4.4))

    _subfig_time_TCNS(FCR_name, ax=fcr, dump_dir=dump_dir,alg_name='GreedyDisDm')
    fcr.legend(bbox_to_anchor=(.35, 1.18, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)
    _subfig_time_TCNS(FCM_name, ax=fcm, dump_dir=dump_dir)
    _subfig_time_TCNS(FUR_name, ax=fur, dump_dir=dump_dir)
    _subfig_time_TCNS(FUM_name, ax=fum, dump_dir=dump_dir)

    fcr.set_title('CR')
    fcm.set_title('CM')
    fur.set_title('UR')
    fum.set_title('UM')

    fig.text(0.5, 0.01, 'Number of users', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=0, h_pad=0.2)
    if network==123:
        plt.savefig(fig_dir +"FnT_greedy_time_123.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir +"FnT_greedy_time_38.pdf", bbox_inches='tight')
if __name__ == "__main__":
    pass
