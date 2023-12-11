#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
Methods for correlation and variable plots
"""
import pickle
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from machine_learning_hep.logger import get_logger

#import matplotlib as mpl
#mpl.use('Agg')

def vardistplot(dfs_input_, mylistvariables_, output_,
                binmin, binmax, plot_options_):
    plot_type_name = "prob_cut_scan"
    plot_options = plot_options_.get(plot_type_name, {}) \
            if isinstance(plot_options_, dict) else {}

    colors = ['r', 'b', 'g']
    figure = plt.figure(figsize=(20, 15))

    figure.suptitle(f"Separation plots for ${binmin} < p_\\mathrm{{T}}/(\\mathrm{{GeV}}/c) < " \
                    f"{binmax}$", fontsize=30)
    for ind, var in enumerate(mylistvariables_, start=1):
        ax = plt.subplot(3, int(len(mylistvariables_)/3+1), ind)
        plt.yscale('log')
        kwargs = {"alpha": 0.3, "density": True, "bins": 100}
        po = plot_options.get(var, {})
        if "xlim" in po:
            kwargs["range"] = (po["xlim"][0], po["xlim"][1])

        for label, color in zip(dfs_input_, colors):
            plt.hist(dfs_input_[label][var], facecolor=color, label=label, **kwargs)

        var_tex = var.replace("_", ":")
        if "xlim" in po:
            plt.xlim(po["xlim"][0], po["xlim"][1])
        if "xlabel" in po:
            var_tex = "$" + po["xlabel"] + "$"
        plt.xlabel(var_tex, fontsize=11)
        plt.ylabel(po.get("ylabel", "entries"), fontsize=11)
        ax.legend()
    plotname = f"{output_}/variablesDistribution_nVar{len(mylistvariables_)}_{binmin}{binmax}.png"
    plt.savefig(plotname, bbox_inches='tight')
    plt.close(figure)

def vardistplot_probscan(dataframe_, mylistvariables_, modelname_, thresharray_, # pylint: disable=too-many-statements
                         output_, suffix_, opt=1, plot_options_=None):

    plot_type_name = "prob_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    figure = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [figure.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

    # Sort the thresharray_
    thresharray_.sort()
    # Re-use skimmed dataframe
    df_skimmed = None
    variables_selected = mylistvariables_ + [f"y_test_prob{modelname_}"]

    xrange_min = []
    xrange_max = []
    ref_hists = []
    for thresh_index, threshold in enumerate(thresharray_):
        selml = f"y_test_prob{modelname_}>{threshold}"
        if df_skimmed is None:
            df_skimmed = dataframe_.query(selml)[variables_selected]
        else:
            df_skimmed = df_skimmed.query(selml)

        for i, var in enumerate(mylistvariables_):

            # Extract minimum and maximum for x-axis, this is only done once
            # for each variable
            if thresh_index == 0:
                axes[i].set_xlabel(var, fontsize=30)
                ylabel = "entries"
                if opt == 1:
                    ylabel += f"/entries(prob{thresharray_[0]})"
                axes[i].set_ylabel(ylabel, fontsize=30)
                axes[i].tick_params(labelsize=20)
                if var in plot_options and "xlim" in plot_options[var]:
                    xrange_min.append(plot_options[var]["xlim"][0])
                    xrange_max.append(plot_options[var]["xlim"][1])
                else:
                    values0 = df_skimmed[var]
                    xrange_min.append(values0.min())
                    xrange_max.append(values0.max())

            n = len(df_skimmed[var])
            lbl = f'prob > {threshold} n = {n}'
            clr = color[thresh_index%len(color)]
            values = df_skimmed[var]
            his, bina = np.histogram(values, range=(xrange_min[i], xrange_max[i]), bins=100)
            if thresh_index == 0:
                ref_hists.append(his)
            width = np.diff(bina)
            center = (bina[:-1] + bina[1:]) / 2

            if opt == 0:
                axes[i].set_yscale('log')
            elif opt == 1:
                his = np.divide(his, ref_hists[i])
                axes[i].set_ylim(0.001, 1.1)
            axes[i].bar(center, his, align='center', width=width, facecolor=clr, label=lbl)
            axes[i].legend(fontsize=10)
    plotname = f"{output_}/variables_distribution_{suffix_}_ratio{opt}.png"
    plt.savefig(plotname, bbox_inches='tight')
    plt.close(figure)

def efficiency_cutscan(dataframe_, mylistvariables_, modelname_, threshold, # pylint: disable=too-many-statements
                       output_, suffix_, plot_options_=None):

    plot_type_name = "eff_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})
    selml = f"y_test_prob{modelname_}>{threshold}"
    dataframe_ = dataframe_.query(selml)

    figure = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [figure.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

    # Available cut options
    cut_options = ["lt", "st", "abslt", "absst"]

    for i, var_tuple in enumerate(mylistvariables_):
        var = var_tuple[0]
        vardir = var_tuple[1]

        axes[i].set_xlabel(var, fontsize=30)
        axes[i].set_ylabel("entries (normalised)", fontsize=30)
        axes[i].tick_params(labelsize=20)
        axes[i].set_yscale('log')
        axes[i].set_ylim(0.1, 1.5)
        values = dataframe_[var].values

        if "abs" in  vardir:
            cen = var_tuple[2] if len(var_tuple) > 2 else None
            if cen is None:
                get_logger().error("Absolute cut chosen for %s. " \
                        "However, no central value provided", var)
                continue
            values = np.array([abs(v - cen) for v in values])

        nbinscan = 100
        minv, maxv = values.min(), values.max()
        if var in plot_options and "xlim" in plot_options[var]:
            minv = plot_options[var]["xlim"][0]
            maxv = plot_options[var]["xlim"][1]
        else:
            minv = values.min()
            maxv = values.max()
        _, bina = np.histogram(values, range=(minv, maxv), bins=nbinscan)
        widthbin = (maxv - minv)/(float)(nbinscan)
        width = np.diff(bina)
        center = (bina[:-1] + bina[1:]) / 2
        den = len(values)
        ratios = deque()

        if vardir not in cut_options:
            get_logger().error("Please choose cut option from %s. " \
                    "Your current setting for variable %s is %s", str(cut_options), vardir, var)
            continue

        if "lt" in vardir:
            for ibin in range(nbinscan):
                values = values[values > minv+widthbin*ibin]
                num = len(values)
                eff = float(num)/float(den)
                ratios.append(eff)
        else:
            for ibin in range(nbinscan, 0, -1):
                values = values[values < minv+widthbin*ibin]
                num = len(values)
                eff = float(num)/float(den)
                ratios.appendleft(eff)
        lbl = f'prob > {threshold}'
        axes[i].bar(center, ratios, align='center', width=width, label=lbl)
        axes[i].legend(fontsize=30)
    plotname = f"{output_}/variables_effscan_prob{threshold}_{suffix_}.png"
    plt.savefig(plotname, bbox_inches='tight')
    plt.close(figure)

def picklesize_cutscan(dataframe_, mylistvariables_, output_, suffix_, plot_options_=None): # pylint: disable=too-many-statements

    plot_type_name = "picklesize_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})

    figure = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [figure.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

    df_reference_pkl_size = len(pickle.dumps(dataframe_, protocol=4))
    df_reference_size = dataframe_.shape[0] * dataframe_.shape[1]

    for i, var_tuple in enumerate(mylistvariables_):
        var = var_tuple[0]
        vardir = var_tuple[1]
        cen = var_tuple[2]

        axes[i].set_xlabel(var, fontsize=30)
        axes[i].set_ylabel("rel. pickle size after cut", fontsize=30)
        axes[i].tick_params(labelsize=20)
        axes[i].set_yscale('log')
        axes[i].set_ylim(0.005, 1.5)
        values = dataframe_[var].values
        if "abs" in  vardir:
            values = np.array([abs(v - cen) for v in values])
        nbinscan = 100
        if var in plot_options and "xlim" in plot_options[var]:
            minv = plot_options[var]["xlim"][0]
            maxv = plot_options[var]["xlim"][1]
        else:
            minv = values.min()
            maxv = values.max()
        _, bina = np.histogram(values, range=(minv, maxv), bins=nbinscan)
        widthbin = (maxv - minv)/(float)(nbinscan)
        width = np.diff(bina)
        center = (bina[:-1] + bina[1:]) / 2
        ratios_df_pkl_size = deque()
        ratios_df_size = deque()
        df_skimmed = dataframe_
        if "lt" in vardir:
            for ibin in range(nbinscan):
                df_skimmed = df_skimmed.iloc[values > minv+widthbin*ibin]
                values = values[values > minv+widthbin*ibin]
                num = len(pickle.dumps(df_skimmed, protocol=4))
                eff = float(num)/float(df_reference_pkl_size)
                ratios_df_pkl_size.append(eff)
                num = df_skimmed.shape[0] * df_skimmed.shape[1]
                eff = float(num)/float(df_reference_size)
                ratios_df_size.append(eff)
        elif "st" in vardir:
            for ibin in range(nbinscan, 0, -1):
                df_skimmed = df_skimmed.iloc[values < minv+widthbin*ibin]
                values = values[values < minv+widthbin*ibin]
                num = len(pickle.dumps(df_skimmed, protocol=4))
                eff = float(num)/float(df_reference_pkl_size)
                ratios_df_pkl_size.appendleft(eff)
                num = df_skimmed.shape[0] * df_skimmed.shape[1]
                eff = float(num)/float(df_reference_size)
                ratios_df_size.appendleft(eff)
        axes[i].bar(center, ratios_df_pkl_size, align='center', width=width, label="rel. pkl size",
                    alpha=0.5)
        axes[i].bar(center, ratios_df_size, align='center', width=width, label="rel. df length",
                    alpha=0.5)
        axes[i].legend(fontsize=30)
    plotname = f"{output_}/variables_cutscan_picklesize_{suffix_}.png"
    plt.savefig(plotname, bbox_inches='tight')
    plt.close(figure)


def scatterplot(dfs_input_, mylistvariablesx_,
                mylistvariablesy_, output_, binmin, binmax):
    colors = ['r', 'b', 'g']
    figurecorr = plt.figure(figsize=(30, 20)) # pylint: disable=unused-variable
    for ind, (var_x, var_y) in enumerate(zip(mylistvariablesx_, mylistvariablesy_), start=1):
        axcorr = plt.subplot(3, int(len(mylistvariablesx_)/3+1), ind)
        plt.xlabel(var_x, fontsize=11)
        plt.ylabel(var_y, fontsize=11)
        title_str = 'Pearson coef. '
        for label, color in zip(dfs_input_, colors):
            plt.scatter(dfs_input_[label][var_x], dfs_input_[label][var_y],
                        alpha=0.4, c=color, label=label)
            pearson = dfs_input_[label].corr(numeric_only=True)[var_x][var_y].round(2)
            title_str += f'{label}: {pearson}, '
        plt.title(title_str)
        axcorr.legend()
    plotname = f"{output_}/variablesScatterPlot{binmin}{binmax}.png"
    plt.savefig(plotname, bbox_inches='tight')
    plt.close(figurecorr)


def correlationmatrix(dataframe, mylistvariables, label, output, binmin, binmax,
                      plot_options_=None):
    corr = dataframe[mylistvariables].corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    _, ax = plt.subplots(figsize=(10, 8))
    #sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
    plot_type_name = "prob_cut_scan"
    plot_options = plot_options_.get(plot_type_name, {}) \
            if isinstance(plot_options_, dict) else {}
    labels = []
    for myvar in mylistvariables:
        if myvar in plot_options and "xlabel" in plot_options[myvar]:
            tex_var = "$" + plot_options[myvar]["xlabel"] + "$"
            labels.append(tex_var)
        else:
            labels.append(myvar.replace("_", ":"))

    if not labels:
        labels = "auto"
    sns.heatmap(corr, mask=mask,
                cmap=sns.diverging_palette(220, 10, as_cmap=True), vmin=-1, vmax=1,
                square=True, ax=ax, xticklabels=labels, yticklabels=labels)
    ax.text(0.7, 0.9, f"${binmin} < p_\\mathrm{{T}}/(\\mathrm{{GeV}}/c) < {binmax}$\n{label}",
            verticalalignment='center', transform=ax.transAxes, fontsize=13)
    plt.savefig(output, bbox_inches='tight')
    plt.close()
