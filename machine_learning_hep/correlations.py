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


def vardistplot(dataframe_sig_, dataframe_bkg_, mylistvariables_, output_):
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
    plotname = output_+'/variablesDistribution.png'
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO


def scatterplot(dataframe_sig_, dataframe_bkg_, mylistvariablesx_, mylistvariablesy_, output_):
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
    plotname = output_+'/variablesScatterPlot.png'
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO


def correlationmatrix(dataframe, output_, label):
    corr = dataframe.corr()
    f, ax = plt.subplots(figsize=(10, 8)) # pylint: disable=unused-variable
    plt.title(label, fontsize=11)
    sns.heatmap(
        corr, mask=np.zeros_like(corr, dtype=np.bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True), vmin=-1, vmax=1,
        square=True, ax=ax, annot=True, fmt=".2f")
    plotname = output_+'/correlationmatrix'+label+'.png'
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO
