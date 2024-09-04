import json, yaml
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
import numpy as np


def LoadJson(file_name):
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


def FormatFig(xlabel,ylabel,ax0,xpos=0.9,ypos=0.9):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=24)
    ax0.set_ylabel(ylabel)

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=14)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    
    import matplotlib.pyplot as plt

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs


def HistRoutine(feed_dict,
                xlabel='',
                ylabel='Normalized number of events',
                reference_name='data',
                logy=False,
                logx = False,
                binning=None,
                label_loc='best',
                plot_ratio=True,
                weights=None,
                #colors from https://github.com/mpetroff/accessible-color-cycles
                color_list = ['black',"#3f90da", "#ffa90e", "#bd1f01",
                              "#94a4a2", "#832db6", "#a96b59", "#e76300",
                              "#b9ac70", "#717581", "#92dadd"],
                ref_plot = {'histtype':'stepfilled','alpha':0.2},
                other_plots = {'histtype':'step','linewidth':2},
                marker_style = 'o',
                uncertainty=None):
    if plot_ratio:
        assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])

    color = {}
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.01),np.quantile(feed_dict[reference_name],0.99),50)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]

    if weights is not None:
        reference_hist,_ = np.histogram(feed_dict[reference_name],weights=weights[reference_name],bins=binning,density=True)
    else:
        reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)

    maxy = 0    
    for ip,plot in enumerate(feed_dict.keys()):
        color[plot] = color_list[ip]
        plot_style = ref_plot if reference_name == plot else other_plots
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,density=True,color=color[plot],
                              weights=weights[plot],**plot_style)
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,density=True,color=color[plot],**plot_style)

        if np.max(dist) > maxy:
            maxy = np.max(dist)
            
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)
                ax1.plot(xaxis,ratio,ms=10,marker = marker_style,color=color[plot],
                         lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')
        ax0.set_ylim(1e-5,10*maxy)
    else:
        ax0.set_ylim(0,1.3*maxy)

    if logx:
        #ax0.set_xscale('log')
        ax1.set_xscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=2)
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Data')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        plt.ylim([0.5,1.5])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        
    return fig,ax0
