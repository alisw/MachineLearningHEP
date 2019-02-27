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
functions for fitting
"""
#import argparse
#import sys
#import os.path

from ROOT import TGraph # pylint: disable=import-error, no-name-in-module

def fitmass(histomass):
    signal = 1
    err_signal = 0.1
    _ = histomass.GetEntries()
    return signal, err_signal

#in this function we have to create a tgraph of the signal yield with errors
def plot_graph_yield(yield_signal, yield_signal_err, binmin, binmax):
    gr = TGraph(0)
    _ = gr.GetN()
    print(yield_signal, yield_signal_err, binmin, binmax)
