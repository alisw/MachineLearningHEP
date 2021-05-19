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
from os.path import join
from collections import deque
from io import BytesIO
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import seaborn as sns

from machine_learning_hep.logger import get_logger

#mpl.use('Agg')

def vardistplot(dataframe_sig_, dataframe_bkg_, mylistvariables_, output_,
                binmin, binmax, plot_options_):
    mpl.rcParams.update({"text.usetex": True})
    plot_type_name = "prob_cut_scan"
    plot_options = plot_options_.get(plot_type_name, {}) \
            if isinstance(plot_options_, dict) else {}


    n_columns = 5
    n_rows = int((len(mylistvariables_)) / n_columns) + 1
    fig_y = 5 * n_rows

    figure = plt.figure(figsize=(20, fig_y))

    i = 1

    for var in mylistvariables_:

        ax = plt.subplot(n_rows, n_columns, i)

        plt.yscale('log')

        po = plot_options.get(var, {})
        kwargs = dict(alpha=0.3, density=True, bins=po.get("bins", 100))
        if "xlim" in po:
            kwargs["range"] = (po["xlim"][0], po["xlim"][1])

        sig_content, sig_edges, _ = plt.hist(dataframe_sig_[var], facecolor='b', label='signal',
                                             **kwargs)
        bkg_content, bkg_edges, _ = plt.hist(dataframe_bkg_[var], facecolor='g', label='background',
                                             **kwargs)
        sig_content = np.array(sig_content) * (sig_edges[1] - sig_edges[0])
        bkg_content = np.array(bkg_content) * (bkg_edges[1] - bkg_edges[0])

        sep_power = sig_content + bkg_content
        ind = sep_power > 0
        sig_content = sig_content[ind]
        bkg_content = bkg_content[ind]

        sep_power = 0.5 * np.sum((sig_content - bkg_content)**2 / (sig_content + bkg_content))

        var_tex = var.replace("_", ":")
        if "xlim" in po:
            plt.xlim(po["xlim"][0], po["xlim"][1])
        if "xlabel" in po:
            var_tex = "$" + po["xlabel"] + "$"

        sep_text = "$p_\\mathrm{sep} = " + f"{sep_power * 100:.4f}" + "\\%$"
        ax.text(0.1, 0.9, sep_text, fontsize=25, transform=ax.transAxes)
        plt.tick_params(labelsize=20)
        plt.xlabel(var_tex, fontsize=30)

        if (i-1) % n_columns == 0:
            plt.ylabel(po.get("ylabel", "entries"), fontsize=30)
        i = i+1
    plotname = output_+'/variablesDistribution_nVar%d_%d%d.png' % \
                            (len(mylistvariables_), binmin, binmax)

    ax = plt.subplot(n_rows, n_columns, i)
    legend_elements = [Patch(facecolor='b', label='signal', alpha=0.3),
                       Patch(facecolor='g', label="background", alpha=0.3)]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=30)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(plotname)
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    mpl.rcParams.update({"text.usetex": False})
    plt.close(figure)
    return imagebytesIO

def vardistplot_probscan(dataframe_, mylistvariables_, modelname_, thresharray_, # pylint: disable=too-many-statements
                         output_, suffix_, opt=1, plot_options_=None):

    plot_type_name = "prob_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    fig = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [fig.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

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
    plotname = join(output_, f"variables_distribution_{suffix_}_ratio{opt}.png")
    plt.savefig(plotname, bbox_inches='tight')

def efficiency_cutscan(dataframe_, mylistvariables_, modelname_, threshold, # pylint: disable=too-many-statements
                       output_, suffix_, plot_options_=None):

    plot_type_name = "eff_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})
    selml = "y_test_prob%s>%s" % (modelname_, threshold)
    dataframe_ = dataframe_.query(selml)

    fig = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [fig.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

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
    plotname = join(output_, f"variables_effscan_prob{threshold}_{suffix_}.png")
    plt.savefig(plotname, bbox_inches='tight')
    plt.savefig(plotname, bbox_inches='tight')

def picklesize_cutscan(dataframe_, mylistvariables_, output_, suffix_, plot_options_=None): # pylint: disable=too-many-statements

    plot_type_name = "picklesize_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})

    fig = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [fig.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

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
    plotname = join(output_, f"variables_cutscan_picklesize_{suffix_}.png")
    plt.savefig(plotname, bbox_inches='tight')

def scatterplot(dataframe_sig_, dataframe_bkg_, mylistvariablesx_,
                mylistvariablesy_, output_, binmin, binmax):
    figurecorr = plt.figure(figsize=(30, 20)) # pylint: disable=unused-variable
    i = 1
    for j, _ in enumerate(mylistvariablesx_):
        axcorr = plt.subplot(3, int(len(mylistvariablesx_)/3+1), i)
        plt.xlabel(mylistvariablesx_[j], fontsize=11)
        plt.ylabel(mylistvariablesy_[j], fontsize=11)
        plt.scatter(
            dataframe_bkg_[mylistvariablesx_[j]], dataframe_bkg_[mylistvariablesy_[j]],
            alpha=0.4, c="g", label="background")
        plt.scatter(
            dataframe_sig_[mylistvariablesx_[j]], dataframe_sig_[mylistvariablesy_[j]],
            alpha=0.4, c="b", label="signal")
        plt.title(
            'Pearson sgn: %s' %
            dataframe_sig_.corr().loc[mylistvariablesx_[j]][mylistvariablesy_[j]].round(2)+
            ',  Pearson bkg: %s' %
            dataframe_bkg_.corr().loc[mylistvariablesx_[j]][mylistvariablesy_[j]].round(2))
        axcorr.legend()
        i = i+1
    plotname = output_+'/variablesScatterPlot%d%d.png' % (binmin, binmax)
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    plt.close(figurecorr)
    return imagebytesIO


def correlationmatrix(dataframe, mylistvariables, output, plot_options_=None):
    corr = dataframe[mylistvariables].corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    _, ax = plt.subplots(figsize=(10, 8))
    mpl.rcParams.update({"text.usetex": True})
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
    ax.tick_params(labelsize=20)
    plt.savefig(output, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    mpl.rcParams.update({"text.usetex": False})
    plt.close()
    return imagebytesIO
