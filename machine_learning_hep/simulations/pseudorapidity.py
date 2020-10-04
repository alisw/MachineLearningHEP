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
import ROOT # pylint: disable=import-error,no-name-in-module
from ROOT import TMath, gROOT

gROOT.SetBatch(True)
# pylint: disable=invalid-name
for pseudo in [0, 1, 1.44, 2, 3, 4, 5]:
    anglerad = 2 * TMath.ATan(TMath.Power(float(TMath.E()), float(-pseudo)))
    print("value at eta=", pseudo, "is %.1f" % (anglerad * 360. / 2. / TMath.Pi()))



radius = 2.2 #cm
halflength = 22.75 #cm
pseudo = - TMath.Log(TMath.Tan(TMath.ATan(2.2/22.75)/2.))
print(pseudo, " pseudounits")
degrees = TMath.ATan(2.2/22.75) *360 / (2 * TMath.Pi())
print(degrees)
