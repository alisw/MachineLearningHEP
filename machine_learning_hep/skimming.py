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
import numba
import numpy as np
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F, TCanvas # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.general import filter_df_cand
from machine_learning_hep.models import apply # pylint: disable=import-error
#from machine_learning_hep.general import get_database_ml_parameters # pylint: disable=import-error


@numba.njit
def selectcandidateml(array_inv_mass, array_prob, probcut):
    array_inv_mass_sel = []
    for i, inv_mass  in enumerate(array_inv_mass):
        prob = array_prob[i]
        if prob > probcut:
            array_inv_mass_sel.append(inv_mass)
    return array_inv_mass_sel

 # pylint: disable=too-many-arguments,too-many-statements
def fill_mass_array(data, namefiledf, namefilehisto, namedfout, var_pt, ptmin, ptmax,
                    useml, modelname, model, probcut, case):

    presel_reco = data[case]["presel_reco"]
    var_training = data[case]["variables"]["var_training"]
    invmassbins = data[case]["invmassbins"]
    invmasslow = data[case]["invmasslow"]
    invmasshigh = data[case]["invmasshigh"]
    var_mass = data[case]["variables"]["var_inv_mass"]
    dirmod = data[case]["output_folders"]["mlout"]

    model = os.path.join(dirmod, model)

    fileinput = open(namefiledf, "rb")
    df = pickle.load(fileinput)
    df = df.astype(np.float)

    array_inv_mass_sel = []
    #print(var_pt)
    h_invmass = TH1F("h_invmass_" + str(ptmin) + "-" + str(ptmax), "", \
                     invmassbins, invmasslow, invmasshigh)
    df = df.query("pt_cand>@ptmin and pt_cand<@ptmax")
    if presel_reco is not None:
        df = df.query(presel_reco)

    if useml == 0:
        df = filter_df_cand(df, data[case], 'sel_std_analysis')
        array_inv_mass_sel = df.loc[:, var_mass].values
        #array_pt = df.loc[:, var_pt].values
        #array_inv_mass_sel = selectcandidate(array_inv_mass, array_pt, ptmin, ptmax)
    if useml == 1:
        df = filter_df_cand(df, data[case], 'presel_track_pid')
        #Apply preselection similar as to trained model
        array_inv_mass = df.loc[:, var_mass].values
        mod = pickle.load(open(model, 'rb'))
        df = apply("BinaryClassification", [modelname], [mod], df, var_training)
        array_prob = df.loc[:, "y_test_prob" + modelname].values
        array_inv_mass_sel = selectcandidateml(array_inv_mass, array_prob, probcut)
        df.to_pickle(namedfout)
    fill_hist(h_invmass, array_inv_mass_sel)
    f = TFile(namefilehisto, "recreate")
    f.cd()
    h_invmass.Write()
    f.Close()

# pylint: disable=too-many-arguments
def create_inv_mass(data, listinput_df, listoutputhisto, listoutputdf, pt_var, ptmin, ptmax,
                    useml, modelname, model, probcut, case):
    processes = [mp.Process(target=fill_mass_array, \
                 args=(data, listinput_df[index], listoutputhisto[index], \
                       listoutputdf[index], pt_var, ptmin, ptmax, \
                       useml, modelname, model, probcut, case))
                 for index, _ in enumerate(listinput_df)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    invmassbins = data[case]["invmassbins"]
    invmasslow = data[case]["invmasslow"]
    invmasshigh = data[case]["invmasshigh"]

    h_invmass_tot = TH1F("h_invmass_tot_" + str(ptmin)+ "-" +str(ptmax), \
                         "", invmassbins, invmasslow, invmasshigh)
    for fileout in listoutputhisto:
        myfile_ = TFile.Open(fileout)
        h_invmass = myfile_.Get("h_invmass_" + str(ptmin) + "-" + str(ptmax))
        h_invmass_tot.Add(h_invmass_tot, h_invmass)
    return h_invmass_tot

def plothisto(hmass, nameoutfile):
    canvas = TCanvas('canvas', '', 200, 10, 700, 900)
    hmass.Draw()
    canvas.SaveAs(nameoutfile)
