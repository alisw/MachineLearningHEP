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

import pandas as pd
import os
import yaml

from ROOT import TFile, TH1F, TCanvas # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.selectionutils import getnormforselevt

case = "Dspp5TeV"

with open("data/database_ml_parameters.yml", 'r') as param_config:
    data_param = yaml.load(param_config)

namefile_evt = data_param[case]["files_names"]["namefile_evt_skim_tot"]
folder = data_param[case]["output_folders"]["pkl_merged"]["mc"]

df_evt_all = pd.read_pickle(os.path.join(folder, namefile_evt))

nselevt = len(df_evt_all.query("is_ev_rej==0"))
norm = getnormforselevt(df_evt_all)

hNorm = TH1F("hEvForNorm",";;Normalisation",2,0.5,2.5)
hNorm.GetXaxis().SetBinLabel(1,"normsalisation factor")
hNorm.GetXaxis().SetBinLabel(2,"selected events")
hNorm.SetBinContent(1,norm)
hNorm.SetBinContent(2,nselevt)

outfile = TFile("Normalisation_%s.root" % case,"recreate")
hNorm.Write()
outfile.Close()

