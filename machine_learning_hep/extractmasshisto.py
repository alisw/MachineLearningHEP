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
import os
import pickle
import pandas as pd
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F, TCanvas # pylint: disable=import-error, no-name-in-module
#from machine_learning_hep.selectionutils import getnormforselevt

# pylint: disable=too-many-statements
def extractmasshisto(data_config, data, case, useml, mcordata, index):

    mass_fit_lim = data[case]['mass_fit_lim']
    bin_width = data[case]['bin_width']
    namefilereco_ml_tot = data[case]["files_names"]["namefile_reco_skim_ml_tot"]
    namefilereco_std_tot = data[case]["files_names"]["namefile_reco_skim_std_tot"]
    namefile_evt_skim_tot = data[case]["files_names"]["namefile_evt_skim_tot"]
    outputdirfin = data[case]["output_folders"]["pkl_final"][mcordata][index]
    modelname = data_config["analysis"]["modelname"]

    namefilereco_ml_tot = os.path.join(outputdirfin, namefilereco_ml_tot)
    namefilereco_std_tot = os.path.join(outputdirfin, namefilereco_std_tot)
    namefile_evt_skim_tot = os.path.join(outputdirfin, namefile_evt_skim_tot)
    probcutoptimal = data_config["analysis"]["probcutoptimal"]
    outputdir = data[case]["output_folders"]["plotsanalysis"][mcordata][index]
    num_bins = (mass_fit_lim[1] - mass_fit_lim[0]) / bin_width
    num_bins = int(round(num_bins))

    analysis_bin_min = data_config['analysis']['analysisbinmin']
    analysis_bin_max = data_config['analysis']['analysisbinmax']
    model_bin_min = data_config['analysis']['modelbinmin']
    model_bin_max = data_config['analysis']['modelbinmax']
    modeltouse = data_config['analysis']['modeltouse']
    usecustomsel = data_config["analysis"][mcordata]["std"]["usecustom"]

    index = 0
    histomassall = []
    for imin, imax in zip(analysis_bin_min, analysis_bin_max):
        filename = namefilereco_ml_tot
        df = pd.DataFrame()
        if useml == 0:
            df = pickle.load(open(namefilereco_std_tot, "rb"))
        if useml == 1:
            binsamplemin = model_bin_min[modeltouse[index]]
            binsamplemax = model_bin_max[modeltouse[index]]
            filename = filename.replace(".pkl", "%d_%d.pkl" \
                                                        % (binsamplemin, binsamplemax))
            df = pickle.load(open(filename, "rb"))
            prob_cut = probcutoptimal[modeltouse[index]]
            sel_ml = "y_test_prob%s>%s" % (modelname, prob_cut)
            df = df.query(sel_ml)
        array_inv_mass_sel = df.inv_mass.values
        namehisto = "h_invmass%d_%d_std_usecust%d" % (imin, imax, (int)(usecustomsel))
        if useml == 1:
            namehisto = "h_invmass%d_%d_ml_prob%.2f" % (imin, imax, prob_cut)
        h_invmass = TH1F(namehisto, "", num_bins, mass_fit_lim[0], mass_fit_lim[1])
        fill_hist(h_invmass, array_inv_mass_sel)
        histomassall.append(h_invmass)
        c = TCanvas('c%d' % index, '', 500, 500)
        h_invmass.Draw()
        c.SaveAs(outputdir + "/" + namehisto+".pdf")
        index = index + 1
    namefile = "masshisto%s_useml%s_usecustom%d.root" % (case, useml, (int)(usecustomsel))
    if useml == 1:
        namefile = "masshisto%s_useml%s_%.2f.root" % (case, useml, \
                                                  probcutoptimal[0])
    myfile = TFile.Open(outputdir + "/" + namefile, "recreate")
    myfile.cd()
    for indexh, _ in enumerate(analysis_bin_min):
        histomassall[indexh].Write()
    h_invmass.Write()
    myfile.Close()
