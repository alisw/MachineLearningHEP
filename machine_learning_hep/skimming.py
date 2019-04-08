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
utilities for skimming ttrees
"""
import os
import multiprocessing as mp
import pickle
import numpy as np
#from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
#from ROOT import TFile, TH1F, TCanvas # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.general import filter_df_cand
from machine_learning_hep.models import apply # pylint: disable=import-error
from machine_learning_hep.selectionutils import selectcandidateml, selectcand_lincut
#from machine_learning_hep.general import get_database_ml_parameters # pylint: disable=import-error

 # pylint: disable=too-many-arguments,too-many-statements
def selectcandidates(data, namefiledf, namefiledf_ml, namefiledf_std, var_pt, ptmin, ptmax,
                     useml, modelname, model, probcut, case, std_cuts_map=None, ibin_std_cuts=None):

    presel_reco = data[case]["presel_reco"]
    var_training = data[case]["variables"]["var_training"]
    dirmod = data[case]["output_folders"]["mlout"]

    model = os.path.join(dirmod, model)

    fileinput = open(namefiledf, "rb")
    df = pickle.load(fileinput)
    df = df.astype(np.float)

    df = df.query("pt_cand>@ptmin and pt_cand<@ptmax")
    _ = "%d%d%s" % (ptmin, ptmax, var_pt)
    if presel_reco is not None:
        df = df.query(presel_reco)

    if useml == 0:
        if std_cuts_map is None:
            df = filter_df_cand(df, data[case], 'sel_std_analysis')
            df.to_pickle(namefiledf_std)
        else:
            #preselection on pid and track vars using bitmap
            df = filter_df_cand(df, data[case], 'presel_track_pid')
            #apply standard cuts from file
            for icutvar in std_cuts_map:
                if icutvar != "var_binning":
                    array_var = df.loc[:, std_cuts_map[icutvar]["name"]].values
                    is_selected = selectcand_lincut(array_var, \
                            std_cuts_map[icutvar]["min"][ibin_std_cuts], \
                            std_cuts_map[icutvar]["max"][ibin_std_cuts], \
                            std_cuts_map[icutvar]["isabsval"])
                    df = df[is_selected]
            df.to_pickle(namefiledf_std)
    elif useml == 1:
        df = filter_df_cand(df, data[case], 'presel_track_pid')
        mod = pickle.load(open(model, 'rb'))
        df = apply("BinaryClassification", [modelname], [mod], df, var_training)
        array_prob = df.loc[:, "y_test_prob" + modelname].values
        is_selected = selectcandidateml(array_prob, probcut)
        df = df[is_selected]
        df.to_pickle(namefiledf_ml)

def selectcandidatesall(data, listdf, listdfout_ml, listdfout_std, pt_var, ptmin, ptmax,
                        useml, modelname, model, probcut, case, std_cuts_map=None, \
                            ibin_std_cuts=None):
    processes = [mp.Process(target=selectcandidates, \
                 args=(data, listdf[index], listdfout_ml[index], \
                       listdfout_std[index], pt_var, ptmin, ptmax, \
                       useml, modelname, model, probcut, case, std_cuts_map, ibin_std_cuts))
                 for index, _ in enumerate(listdf)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
