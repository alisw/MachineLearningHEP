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
Script containing all helper functions related to plotting with ROOT

Script also contains the "class Errors", used for systematic uncertainties (to
replace AliHFSystErr from AliPhysics).
"""
# pylint: disable=too-many-lines
# pylint: disable=import-error, no-name-in-module
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TH2F, TH1F
from machine_learning_hep.utilities_plot import fill2dhist
from machine_learning_hep.bitwise import filter_bit_df

def fillvalidationvsmult(dfevt, dfevtevtsel, df_reco):
    """
    Create a TH3F histogram and fill it with three variables from a dataframe.
    """
    _ = len(df_reco)
    hlist = []
    def makenfill(nx, xl, xh, ny, yl, yh, X, Y):
      h = TH2F("h_" + X + "_vs_" + Y, " ; " + X + " ; " + Y, nx, xl, xh, ny, yl, yh)
      fill2dhist(dfevtevtsel, h, X, Y)
      hlist.append(h)
      return h

    makenfill(200, -0.5, 199.5, 1500, -0.5, 1499.5, "n_tracklets", "v0m")
    makenfill(200, -0.5, 199.5, 1500, -0.5, 1499.5, "n_tracklets", "v0m_eq")
    makenfill(200, -0.5, 199.5, 1500, -0.5, 1499.5, "n_tracklets", "v0m_corr")
    makenfill(200, -0.5, 199.5, 1500, -0.5, 1499.5, "n_tracklets", "v0m_eq_corr")

    makenfill(200, -0.5, 199.5, 1000, -0.5, 99.5, "n_tracklets", "perc_v0m")

    makenfill(1500, -0.5, 1499.5, 1000, -0.5, 99.5, "v0m", "perc_v0m")
    makenfill(1500, -0.5, 1499.5, 1000, -0.5, 99.5, "v0m_eq", "perc_v0m")
    makenfill(1500, -0.5, 1499.5, 1000, -0.5, 99.5, "v0m_corr", "perc_v0m")
    makenfill(1500, -0.5, 1499.5, 1000, -0.5, 99.5, "v0m_eq_corr", "perc_v0m")

    hntrklvstrklcorr = TH2F("hntrklvstrklcorr", " ; n_tracklets ; n_tracklets_corr",
                            200, -0.5, 199.5, 200, -0.5, 199.5)
    fill2dhist(dfevtevtsel, hntrklvstrklcorr, "n_tracklets", "n_tracklets_corr")
    hntrklcorrvszvtx = TH2F("hntrklcorrvszvtx", " ; z_vtx ; n_tracklets_corr",
                            100, -15., 15, 200, -0.5, 199.5)
    fill2dhist(dfevtevtsel, hntrklcorrvszvtx, "z_vtx_reco", "n_tracklets_corr")
    hntrklvszvtx = TH2F("hntrklvszvtx", " ; z_vtx ; n_tracklets",
                        100, -15., 15, 200, -0.5, 199.5)
    fill2dhist(dfevtevtsel, hntrklvszvtx, "z_vtx_reco", "n_tracklets")

    hntrklcorrsel = TH1F("hntrklcorrsel", ' ; ntracklets_corr ; Entries',
                         200, -0.5, 199.5)
    hntrklcorrselevtspd = TH1F("hntrklcorrselevtspd", ' ; ntracklets_corr ; Entries',
                               200, -0.5, 199.5)
    hntrklcorrpileup = TH1F("hntrklcorrpileup", ' ; ntracklets_corr ; Entries',
                            200, -0.5, 199.5)
    df_pileup = filter_bit_df(dfevt, 'is_ev_rej', [[4], []])
    df_selevtspd = dfevtevtsel.query("is_ev_sel_shm == 1")

    fill_hist(hntrklcorrsel, dfevtevtsel["n_tracklets_corr"])
    fill_hist(hntrklcorrselevtspd, df_selevtspd["n_tracklets_corr"])
    fill_hist(hntrklcorrpileup, df_pileup["n_tracklets_corr"])
    return hlist + [hntrklvstrklcorr, hntrklcorrvszvtx, hntrklvszvtx, hntrklcorrsel,
            hntrklcorrpileup, hntrklcorrselevtspd]
