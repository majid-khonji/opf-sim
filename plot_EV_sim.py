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
def _subfig_obj(filename, ax, dump_dir='results/dump',color='b',y_lim=None, start_n=100, step_n=100, max_n=3500, set_y_ticks=True, y_tick_step=1000, y_tick_max=6500, darkcolor='dodgerblue', lightcolor='lightblue'):

    x = range(start_n, max_n + 1 , step_n)

    f = np.load(dump_dir + '/' + filename+ ".npz")
    opt_data = f['frac_EV_obj'].tolist()

    f = np.load(dump_dir + '/' + filename+ ".npz")
    alg_data = f['round_EV_obj'].tolist()


    if color == 'b':
        __box_plot(ax, alg_data, name='Greedy-SSP', darkcolor='dodgerblue', lightcolor='lightblue')
    else:
        __box_plot(ax, alg_data, name='PTAS-A-EVSP', darkcolor='green', lightcolor='lightgreen')
    # __box_plot(ax, no_LP_alg_data, name='No LP', darkcolor='green', lightcolor='lightgreen')
    __box_plot(ax, opt_data, name='Gurobi Frac.',darkcolor='darkred',lightcolor='indianred')

    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    _format_exponent(ax, 'y')
    plt.setp(ax, xticks=range(1,len(x)+1, 2))
    ax.set_xticklabels(range(start_n,max_n+1,2*step_n))
    if y_lim != None:
        ax.set_ylim(y_lim)

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # major_ticks = np.arange(0, 101, 20)
    # minor_ticks = np.arange(start_n, max_n+1, step_n)
    # ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    if set_y_ticks:
        y_major_ticks = np.arange(0, y_tick_max, y_tick_step)

        ax.set_yticks(y_major_ticks)
        # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    # ax.grid(which='both')

    # Or if you want different settings for the grids:
    # ax.grid(which='minor', alpha=0.2)
    # ax.grid(which='major', alpha=0.5)


def plot_obj(fixed=True, dump_dir="results/dump/", fig_dir="results/"):
    plt.ioff()

    fig, (L, Q) = plt.subplots(1, 2, sharex='col', figsize=(7, 2.50))
    if fixed:
        L_name = 'EV:[L]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
        Q_name = 'EV:[Q]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
        c='g'
        _subfig_obj(filename=L_name, ax=L, dump_dir=dump_dir, max_n = 1000, start_n=100, step_n=50, set_y_ticks=True, y_tick_max=4200, y_tick_step=800,color=c)
        _subfig_obj(filename=Q_name, ax=Q, dump_dir=dump_dir, max_n = 1000, start_n=100, step_n=50, set_y_ticks=True, y_tick_max=40000, y_tick_step=7000, color=c)
    else:
        L_name = 'EV:[L]__alg4_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
        Q_name = 'EV:[Q]__alg4_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
        c= 'b'
        _subfig_obj(filename=L_name, ax=L, dump_dir=dump_dir, max_n = 1000, start_n=100, step_n=50, set_y_ticks=True, y_tick_max=4200, y_tick_step=800,color=c)
        _subfig_obj(filename=Q_name, ax=Q, dump_dir=dump_dir, max_n = 1000, start_n=100, step_n=50, set_y_ticks=True, y_tick_max=40000, y_tick_step=7000, color=c)

    L.legend(bbox_to_anchor=(.4, 1.20, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)







    L.set_title('Linear Utility')
    Q.set_title('Quadratic Utility')


    fig.text(0.5, 0.01, 'Number of EVs', ha='center', va='center', fontsize=14)

    fig.text(0.00, 0.5, 'Objective Value', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1, w_pad=0.2, h_pad=.2)
    if fixed:
        plt.savefig(fig_dir +"EV_fixed_obj.pdf", bbox_inches='tight')
    else:
        plt.savefig(fig_dir +"EV_obj.pdf", bbox_inches='tight')





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

def _subfig_scenarios(data, ax, num_edges=1,cheat_value = 1, percentage=True, field='round_OPF_frac_comp_count', dump_dir='results/dump', y_lim=None, start_n=100, step_n=100, max_n=3500):

    x = range(start_n, max_n + 1 , step_n)

    f = np.load(dump_dir + '/' + data['L']+ ".npz")


    L_data = (cheat_value*f[field]/float(num_edges)).tolist()

    f = np.load(dump_dir + '/' + data['Q']+ ".npz")
    Q_data = (cheat_value*f[field]/float(num_edges)).tolist()


    __box_plot(ax, L_data, name='Linear Utility', darkcolor='darkred', lightcolor='indianred')
    __box_plot(ax, Q_data, name='Quadratic Utility', darkcolor='dodgerblue', lightcolor='lightblue')
    # __box_plot(ax, fur_data, name='UR',darkcolor='darkgreen',lightcolor='lightgreen')
    # __box_plot(ax, fum_data, name='UM',darkcolor='darkred',lightcolor='indianred')



    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    if percentage:
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    xticks = range(1,len(x)+1, 2)
    plt.setp(ax, xticks=xticks)
    ax.set_xticklabels(range(start_n,max_n+1,2*step_n))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    _format_exponent(ax, 'y')

    # xticks2 = range(0,len(x)+2, 2)
    # ax.plot(xticks2, np.ones(len(xticks2)), color='black',
    #         linewidth=2, linestyle='-')

    if y_lim != None:
        ax.set_ylim(y_lim)

def plot_frac_comp_count(dump_dir="results/dump/", y_label='Percentage of fractional\n components over # Cons.',  fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (ax1) = plt.subplots(1, 1, sharex='col', figsize=(4, 2.5))

    L_name = 'EV:[L]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    Q_name = 'EV:[Q]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    data = {'L': L_name, 'Q':Q_name}
    _subfig_scenarios(data, ax=ax1, percentage=False, field='round_EV_frac_com_percentage',  dump_dir=dump_dir, start_n=100, step_n=50, max_n=1000)
    ax1.set_title("PTAS-A-EVSP")
    ax1.legend(bbox_to_anchor=(.35, 1.21, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)


    fig.text(0.5, 0.01, 'Number of EVs', ha='center', va='center', fontsize=14)
    fig.text(-0.02, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir +"EV_frac_com_alg1.pdf", bbox_inches='tight')

def plot_ar(fixed=True, dump_dir="results/dump/", y_label='Approximation Ratio', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (ax1,ax2) = plt.subplots(1, 2, sharex='col', figsize=(7, 2.5))

    L_name = 'EV:[L]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    Q_name = 'EV:[Q]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    data = {'L': L_name, 'Q':Q_name}
    _subfig_scenarios(data, ax=ax1, percentage=False, field='round_EV_ar', y_lim=[0, 1.1], dump_dir=dump_dir, start_n=100, step_n=50, max_n=1000)
    ax1.set_title("PTAS-A-EVSP")
    ax1.legend(bbox_to_anchor=(.35, 1.21, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    L_name = 'EV:[L]__alg4_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    Q_name = 'EV:[Q]__alg4_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    data = {'L': L_name, 'Q':Q_name}
    _subfig_scenarios(data, ax=ax2, percentage=False, field='round_EV_ar', y_lim=[0, 1.1], dump_dir=dump_dir, start_n=100, step_n=50, max_n=1000)
    ax2.set_title("Greedy-SSP")


    fig.text(0.5, 0.01, 'Number of EVs', ha='center', va='center', fontsize=14)
    fig.text(-0.02, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir +"EV_ar_both.pdf", bbox_inches='tight')

def plot_time(fixed=True, dump_dir="results/dump/", y_label='Running Time (sec.)', fig_dir="results/"):
    plt.ioff()
    # plt.clf()

    fig, (ax1,ax2) = plt.subplots(1, 2, sharex='col', figsize=(7, 2.5))

    L_name = 'EV:[L]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    Q_name = 'EV:[Q]__fixed_int_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    data = {'L': L_name, 'Q':Q_name}
    _subfig_scenarios(data, ax=ax1, cheat_value=20, percentage=False, field='round_EV_time', y_lim=None, dump_dir=dump_dir, start_n=100, step_n=50, max_n=1000)
    ax1.set_title("PTAS-A-EVSP")
    ax1.legend(bbox_to_anchor=(.35, 1.21, 0, 0), loc=3, ncol=4, borderaxespad=0., fontsize=12)

    L_name = 'EV:[L]__alg4_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    Q_name = 'EV:[Q]__alg4_max_n=1000_step_n=50_start_n=100_reps=40_capacity=1000000'
    data = {'L': L_name, 'Q':Q_name}
    _subfig_scenarios(data, ax=ax2, percentage=False, field='round_EV_time', y_lim=[1,8], dump_dir=dump_dir, start_n=100, step_n=50, max_n=1000)
    ax2.set_title("Greedy-SSP")


    fig.text(0.5, 0.01, 'Number of EVs', ha='center', va='center', fontsize=14)
    fig.text(-0.02, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=14)


    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir +"EV_time_both.pdf", bbox_inches='tight')

def plot_base_load(fixed = True, fig_dir="results/"):
    # plt.ioff()
    day1 = np.array(
        [0.852, 0.788, 0.731, 0.684, 0.649, 0.659, 0.685, 0.755, 0.811, 0.861, 0.873, 0.886, 0.871, 0.833, 0.830, 0.815,
         0.876, 1.042, 1.093, 1.111, 1.100, 1.063, 0.949, 0.842
         ])
    day2 = np.array(
        [0.766, 0.701, 0.658, 0.633, 0.619, 0.627, 0.666, 0.720, 0.806, 0.850, 0.878, 0.919, 0.961, 0.978, 1.009, 1.045,
         1.099, 1.227, 1.251, 1.252, 1.211, 1.131, 0.992, 0.826
         ])

    # 1/3/2011 - 1/4/2011
    day1 = np.array([0.717,    0.637,    0.619,    0.593,    0.604,    0.654,    0.747,    0.809,    0.798,    0.778,    0.782,    0.804,    0.828,    0.788,    0.760,    0.767,    0.843,    1.083,    1.173,    1.200,    1.170,    1.099,    0.961,    0.815])
    day2 = np.array([0.713,    0.657,    0.630,    0.599,    0.616,    0.676,    0.773,    0.827,    0.763,    0.720,    0.709,    0.679,    0.691,    0.676,    0.664,    0.658,    0.738,    0.972,    1.076,    1.118,    1.110,    1.049,    0.908,    0.785])
    two_days = np.append(day1, day2)

    # an instance of 1000 EVs
    # alg 3
    power=np.array([     0.,      0.,      0.,      0.,  13250.,  42500.,  96750.,
       132375., 194625., 276250., 410000., 395000., 404000., 414500.,
       402000., 411500., 334875., 293375., 259625., 253250., 273625.,
       299125., 369000., 428875., 486375., 496875., 551750., 570250.,
       562750., 511500., 441125., 311250., 325750., 117375.,  68375.,
         8500.,   5250.,      0.,      0.,      0.,      0.,      0.,
            0.,      0.,      0.,      0.,      0.,      0.])

    powerf = np.array([     0.        ,      0.        ,   1750.        ,   3500.        ,
        32671.18755107,  76082.76774459, 126029.49077641, 274267.24961154,
       303403.3096745 , 440350.        , 432550.        , 424100.        ,
       433850.        , 458550.        , 460500.        , 470250.        ,
       430600.        , 322700.        , 289550.        , 277850.        ,
       285000.        , 309050.        , 383150.        , 452700.        ,
       502100.        , 544350.        , 572300.        , 588550.        ,
       597650.        , 592450.        , 567100.        , 532000.        ,
       476100.        , 447500.        , 378170.3664934 , 339964.30762589,
        45244.27630058,  26831.85874216,    750.        ,      0.        ,
            0.        ,      0.        ,      0.        ,      0.        ,
            0.        ,      0.        ,      0.        ,      0.        ])
    # alg 4
    power = np.array([     0.,      0.,   1750.,   7000.,  19500.,  49750.,  48000.,
       140750., 165750., 312250., 435500., 425000., 400000., 425000.,
       425000., 425000., 363625., 164375., 185000., 156625., 201875.,
       188875., 318875., 347250., 377375., 418125., 455500., 534875.,
       505250., 477625., 391625., 363500., 428250., 178250., 217750.,
        87500.,      0.,      0.,      0.,      0.,      0.,      0.,
            0.,      0.,      0.,      0.,      0.,      0.])

    powerf = np.array([     0.        ,      0.        ,   1750.        ,   7000.        ,
        15344.66164853,  45811.12525463,  32011.25716534, 114889.21252895,
       171075.87162592, 278697.70299428, 490659.59106258, 477400.        ,
       461800.        , 487800.        , 506000.        , 501450.        ,
       452050.        , 296050.        , 237550.        , 220000.        ,
       239500.        , 285650.        , 375350.        , 470250.        ,
       536550.        , 572950.        , 590500.        , 610650.        ,
       599600.        , 560600.        , 497550.        , 462450.        ,
       452590.50648718, 180306.61291437, 166114.86370025,  50326.62850272,
            0.        ,      0.        ,      0.        ,      0.        ,
            0.        ,      0.        ,      0.        ,      0.        ,
            0.        ,      0.        ,      0.        ,      0.        ])

    power_fix = np.array([     0.,      0.,      0.,      0.,      0.,      0.,      0.,
            0.,      0.,      0.,      0.,      0.,      0.,      0.,
        74750., 388000., 413125., 270625., 195375., 178875., 232500.,
       266375., 278000., 265750., 251750., 250000., 244750., 243000.,
       243000., 242250., 236250., 215625., 187125., 152625., 126375.,
       104250.,  77625.,  57750.,  42750.,  32250.,  19125.,  13875.,
        11625.,   6000.,   4500.,   4500.,   1875.,   1500.])

    powerf_fix = np.array([     0. ,      0. ,      0. ,      0. ,      0. ,      0. ,
            0. ,      0. ,      0. ,      0. ,      0. ,      0. ,
            0. ,      0. ,  74750. , 403725. , 452050. , 296050. ,
       237550. , 220000. , 239500. , 280487.5, 279000. , 266750. ,
       252750. , 251000. , 245750. , 244000. , 244000. , 243250. ,
       237250. , 216625. , 188125. , 153625. , 127375. , 104500. ,
        77625. ,  57750. ,  42750. ,  32250. ,  19125. ,  13875. ,
        11625. ,   6000. ,   4500. ,   4500. ,   1875. ,   1500. ])


    fig, (ax) = plt.subplots(1, 1, sharex='col', figsize=(7, 2.0))



    p0,=ax.plot(650*two_days, label = "Base Load", color='darkgray')
    p4,=ax.plot(650*two_days+powerf/1000., label="Gurobi Frac. (SSP)",color='darkred', alpha=.7)
    p1,= ax.plot(650*two_days+power_fix/1000., label="PTAS-A-EVSP", color='green')
    p3,=ax.plot(650*two_days+powerf_fix/1000., label="Gurobi Frac. (A-EVSP)",color='coral',alpha=.7)
    p2,=ax.plot(650*two_days+power/1000., label="Greedy-SSP", color='dodgerblue')
    # ax.plot(650*two_days+powero/1000., label="OPT EV Charge + Base Load")

    plt.axvline(x=18,linestyle="--", color='darkgray', linewidth=1)
    ax.legend(bbox_to_anchor=(.00, 1.05, 0, 0), loc=3, ncol=3, borderaxespad=0., fontsize=11.5)

    ax.grid(True)
    major_ticks = np.arange(0, 49, 2)
    ax.set_xticks(major_ticks)
    for tick in ax.get_xticklabels():
        # tick.set_rotation(30)
        tick.set_fontsize(8)

    fig.text(0.5, 0.00, 'Time (hour)', ha='center', va='center', fontsize=14)
    fig.text(0.00, 0.5, "Total Power (kW)", ha='center', va='center', rotation='vertical', fontsize=14)
    # minor_ticks = np.arange(start_n, max_n+1, step_n)
    # ax.set_xticks(minor_ticks, minor=True)
    # y_major_ticks = np.arange(0, y_tick_max, y_tick_step)

    # ax.set_yticks(y_major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    # ax.grid(which='both')

    # Or if you want different settings for the grids:
    # ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


    plt.tight_layout(pad=1, w_pad=.2, h_pad=0.2)
    plt.savefig(fig_dir +"EV_load.pdf", bbox_inches='tight')


if __name__ == "__main__":
    pass
