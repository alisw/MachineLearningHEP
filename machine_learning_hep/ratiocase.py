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
main script for doing final stage analysis
"""
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend
from ROOT import gROOT
from ROOT import TStyle

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
def ratiocase(imult, case_num, typean_num, case_den, typean_den):

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)

    ccross = TCanvas('cCross', 'The Fit Canvas', 100, 600)

    with open("data/database_ml_parameters_%s.yml" % case_num, 'r') as param_config_num:
        data_param_num = yaml.load(param_config_num, Loader=yaml.FullLoader)

    with open("data/database_ml_parameters_%s.yml" % case_den, 'r') as param_config_den:
        data_param_den = yaml.load(param_config_den, Loader=yaml.FullLoader)

    folder_num = data_param_num[case_num]["analysis"][typean_num]["data"]["resultsallp"]
    folder_den = data_param_den[case_den]["analysis"][typean_den]["data"]["resultsallp"]

    binsmin_num = data_param_num[case_num]["analysis"][typean_num]["sel_binmin2"]
    binsmax_num = data_param_num[case_num]["analysis"][typean_num]["sel_binmax2"]

    binmin = binsmin_num[imult]
    binmax = binsmax_num[imult]

    file_num = TFile.Open("%s/finalcross%s%s.root" % (folder_num, case_num, typean_num))
    file_den = TFile.Open("%s/finalcross%s%s.root" % (folder_den, case_den, typean_den))
    print("ratiocase")
    ccross = TCanvas('cCross', 'The Fit Canvas')
    ccross.SetCanvasSize(1500, 1500)
    ccross.SetWindowSize(500, 500)
    ccross.SetLogx()
    hcross_num = file_num.Get("hcross%d" % (imult))
    hcross_den = file_den.Get("hcross%d" % (imult))
    hcross_num.Divide(hcross_den)
    hcross_num.GetXaxis().SetTitle("p_{T} (GeV)")
    hcross_num.GetYaxis().SetTitle("Particle ratio")
    hcross_num.GetYaxis().SetRangeUser(0., 1.)
    hcross_num.Draw()
    ccross.SaveAs("ComparisonRatios%.2f_%.2f_%s%s_%s%s.eps" % \
                  (binmin, binmax, case_num, typean_num, case_den, typean_den))


ratiocase(0, "LcpK0spp", "MBvspt", "D0pp", "MBvspt")
ratiocase(1, "LcpK0spp", "MBvspt", "D0pp", "MBvspt")
ratiocase(2, "LcpK0spp", "MBvspt", "D0pp", "MBvspt")
ratiocase(3, "LcpK0spp", "MBvspt", "D0pp", "MBvspt")

ratiocase(0, "LcpK0spp", "SPDvspt", "D0pp", "SPDvspt")
ratiocase(1, "LcpK0spp", "SPDvspt", "D0pp", "SPDvspt")
ratiocase(2, "LcpK0spp", "SPDvspt", "D0pp", "SPDvspt")

ratiocase(0, "LcpK0spp", "V0mvspt", "D0pp", "V0mvspt")
ratiocase(1, "LcpK0spp", "V0mvspt", "D0pp", "V0mvspt")
ratiocase(2, "LcpK0spp", "V0mvspt", "D0pp", "V0mvspt")
