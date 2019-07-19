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

# pylint: disable=import-error, no-name-in-module, unused-import

def ratio():
    ccross = TCanvas('cCross', 'The Fit Canvas')
    fileoutcrossd0pp = TFile.Open("finalcrossD0pp.root")
    fileoutcrossdspp = TFile.Open("finalcrossDspp.root")
    fileoutcrossLcpkpipp = TFile.Open("finalcrossLcpKpipp.root")
    fileoutcrossLcpk0s = TFile.Open("finalcrossLcpK0spp.root")

    with open("data/database_ml_parameters_D0pp.yml", 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)

    nbins = len(data_param["D0pp"]["analysis"]["sel_binmax2"])
    ccross = TCanvas('cCross', 'The Fit Canvas')
    for imult in range(1,nbins):
        hcrossD0pp = fileoutcrossd0pp.Get("hcross%d" % (imult))
        hcrossDspp = fileoutcrossdspp.Get("hcross%d" % (imult))
        hcrossLcpKpipp = fileoutcrossLcpkpipp.Get("hcross%d" % (imult))
        hcrossLcpK0spp = fileoutcrossLcpk0s.Get("hcross%d" % (imult))
        hcrossLcpKpipp.Divide(hcrossD0pp)
        hcrossLcpK0spp.Divide(hcrossD0pp)
        hcrossDspp.Divide(hcrossD0pp)
        hcrossLcpKpipp.SetLineColor(imult+1)
        hcrossDspp.SetLineColor(imult+1)
        hcrossLcpK0spp.SetLineColor(imult+1)
        hcrossLcpKpipp.SetMinimum(0.)
        hcrossLcpKpipp.SetMaximum(1.4)
        hcrossLcpKpipp.GetYaxis().SetRangeUser(0., 1.5)
        hcrossLcpKpipp.Draw("same")
        hcrossLcpK0spp.Draw("same")
        hcrossDspp.Draw("same")
    ccross.SaveAs("ComparisonRatios.pdf")
ratio()
