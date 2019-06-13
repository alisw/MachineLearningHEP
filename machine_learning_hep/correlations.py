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
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def vardistplot(dataframe_sig_, dataframe_bkg_, mylistvariables_, output_,
                binmin, binmax):
    figure = plt.figure(figsize=(20, 15)) # pylint: disable=unused-variable
    i = 1
    for var in mylistvariables_:
        ax = plt.subplot(3, int(len(mylistvariables_)/3+1), i)
        plt.xlabel(var, fontsize=11)
        plt.ylabel("entries", fontsize=11)
        plt.yscale('log')
        kwargs = dict(alpha=0.3, density=True, bins=100)
        plt.hist(dataframe_sig_[var], facecolor='b', label='signal', **kwargs)
        plt.hist(dataframe_bkg_[var], facecolor='g', label='background', **kwargs)
        ax.legend()
        i = i+1
    plotname = output_+'/variablesDistribution%d%d.png' % (binmin, binmax)
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO

def vardistplot_probscan(dataframe_, mylistvariables_, modelname_, tresharray_,
                         output_, suffix_, opt = 1):
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    dfprob = []
    list_ns_th_var = []
    list_bins_th_var = []

    for treshold in tresharray_:
        selml = "y_test_prob%s>%s" % (modelname_, treshold)
        df_ = dataframe_.query(selml)
        dfprob.append(df_)

    figure, ax = plt.subplots(figsize=(60, 25)) # pylint: disable=unused-variable
    i = 1
    for var in mylistvariables_:
        list_ns_th = []
        list_bins_th = []
        isvarpid = "TPC" in var or "TOF" in var

        ax = plt.subplot(3, int(len(mylistvariables_)/3+1), i)
        plt.xlabel(var, fontsize=30)
        plt.ylabel("entries", fontsize=30)
        plt.yscale('log')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        j = 0
        values0 = dfprob[0][var]
        minv, maxv = values0.min(), values0.max()
        if isvarpid is True:
            minv, maxv = -4, 4
        his0, _ = np.histogram(dfprob[0][var], range=(minv, maxv), bins=100)
        for treshold in tresharray_:
            n = len(dfprob[j][var])
            text = f'prob > {treshold} n = {n}'
            lbl = text
            clr = color[j]
            values = dfprob[j][var]
            his, bina = np.histogram(values, range=(minv, maxv), bins=100)
            width = np.diff(bina)
            center = (bina[:-1] + bina[1:]) / 2
            if opt == 0:
                ax.bar(center, his, align='center', width=width, facecolor=clr, label=lbl)
            if opt == 1:
                ratio = np.divide(his,his0)
                ax.bar(center, ratio, align='center', width=width, facecolor=clr, label=lbl)
                plt.ylim(0.001,10)
            j = j+1
        ax.legend(fontsize=10)
        i = i+1
    plotname = output_+'/variablesDistribution_'+suffix_+'.png'
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
    plotname = output_+'/variablesScatterPlot%f%f.png' % (binmin, binmax)
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO


def correlationmatrix(dataframe, output_, label, binmin, binmax):
    corr = dataframe.corr()
    f, ax = plt.subplots(figsize=(10, 8)) # pylint: disable=unused-variable
    plt.title(label, fontsize=11)
    sns.heatmap(
        corr, mask=np.zeros_like(corr, dtype=np.bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True), vmin=-1, vmax=1,
        square=True, ax=ax)
    plotname = output_+'/correlationmatrix%f%f.png' % (binmin, binmax)
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO
