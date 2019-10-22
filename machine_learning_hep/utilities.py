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
main script for doing data processing, machine learning and analysis
"""
import pickle
import bz2
import gzip
import lzma
import os
import numpy as np
import pandas as pd
import lz4
import math
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TH1F, TH2F, TLatex, TGraphAsymmErrors  # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.selectionutils import select_runs
from array import *
def openfile(filename, attr):
    if filename.lower().endswith('.bz2'):
        return bz2.BZ2File(filename, attr)
    if filename.lower().endswith('.xz'):
        return lzma.open(filename, attr)
    if filename.lower().endswith('.gz'):
        return gzip.open(filename, attr)
    if filename.lower().endswith('.lz4'):
        return lz4.frame.open(filename, attr)
    return open(filename, attr)

def selectdfquery(dfr, selection):
    if selection is not None:
        dfr = dfr.query(selection)
    return dfr

def selectdfrunlist(dfr, runlist, runvar):
    if runlist is not None:
        isgoodrun = select_runs(runlist, dfr[runvar].values)
        dfr = dfr[np.array(isgoodrun, dtype=bool)]
    return dfr

def merge_method(listfiles, namemerged):
    dflist = []
    for myfilename in listfiles:
        myfile = openfile(myfilename, "rb")
        df = pickle.load(myfile)
        dflist.append(df)
    dftot = pd.concat(dflist)
    pickle.dump(dftot, openfile(namemerged, "wb"), protocol=4)

# pylint: disable=too-many-nested-blocks
def list_folders(main_dir, filenameinput, maxfiles):
    if not os.path.isdir(main_dir):
        print("the input directory =", main_dir, "doesnt exist")
    list_subdir0 = os.listdir(main_dir)
    listfolders = list()
    for subdir0 in list_subdir0:
        subdir0full = os.path.join(main_dir, subdir0)
        if os.path.isdir(subdir0full):
            list_subdir1 = os.listdir(subdir0full)
            for subdir1 in list_subdir1:
                subdir1full = os.path.join(subdir0full, subdir1)
                if os.path.isdir(subdir1full):
                    list_files_ = os.listdir(subdir1full)
                    for myfile in list_files_:
                        filefull = os.path.join(subdir1full, myfile)
                        if os.path.isfile(filefull) and \
                        myfile == filenameinput:
                            listfolders.append(os.path.join(subdir0, subdir1))
    if maxfiles is not -1:
        listfolders = listfolders[:maxfiles]
    return  listfolders

def create_folder_struc(maindir, listpath):
    for path in listpath:
        path = path.split("/")

        folder = os.path.join(maindir, path[0])
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(folder, path[1])
        if not os.path.exists(folder):
            os.makedirs(folder)

def checkdirlist(dirlist):
    exfolders = 0
    for _, mydir in enumerate(dirlist):
        if os.path.exists(mydir):
            print("rm -rf ", mydir)
            exfolders = exfolders - 1
    return exfolders

def checkdir(mydir):
    exfolders = 0
    if os.path.exists(mydir):
        print("rm -rf ", mydir)
        exfolders = -1
    return exfolders

def checkmakedirlist(dirlist):
    for _, mydir in enumerate(dirlist):
        print("creating folder ", mydir)
        os.makedirs(mydir)

def checkmakedir(mydir):
    print("creating folder ", mydir)
    os.makedirs(mydir)

def appendfiletolist(mylist, namefile):
    return [os.path.join(path, namefile) for path in mylist]

def appendmainfoldertolist(prefolder, mylist):
    return [os.path.join(prefolder, path) for path in mylist]

def createlist(prefolder, mylistfolder, namefile):
    listfiles = appendfiletolist(mylistfolder, namefile)
    listfiles = appendmainfoldertolist(prefolder, listfiles)
    return listfiles
def seldf_singlevar(dataframe, var, minval, maxval):
    dataframe = dataframe.loc[(dataframe[var] > minval) & (dataframe[var] < maxval)]
    return dataframe

def split_df_sigbkg(dataframe_, var_signal_):
    dataframe_sig_ = dataframe_.loc[dataframe_[var_signal_] == 1]
    dataframe_bkg_ = dataframe_.loc[dataframe_[var_signal_] == 0]
    return dataframe_sig_, dataframe_bkg_

def createstringselection(var, low, high):
    string_selection = "dfselection_"+(("%s_%.1f_%.1f") % (var, low, high))
    return string_selection

def mergerootfiles(listfiles, mergedfile):
    outstring = ""
    for indexp, _ in enumerate(listfiles):
        outstring = outstring + listfiles[indexp] + " "
    os.system("hadd -f %s  %s " % (mergedfile, outstring))

def createhisto(stringname, nbins, rmin, rmax):
    hden = TH1F("hden" + stringname, "hden" + stringname, nbins, rmin, rmax)
    hnum = TH1F("hnum" + stringname, "hnum" + stringname, nbins, rmin, rmax)
    hnum.Sumw2()
    hden.Sumw2()
    return hden, hnum

def makeff(dfevt, selnum, selden, stringname, nbins, rmin, rmax, variable):
    #loadstyle()
    hden, hnum = createhisto(stringname, nbins, rmin, rmax)
    dnum = dfevt
    dden = dfevt
    if selnum is not None:
        dnum = dfevt.query(selnum)
    if selden is not None:
        dden = dfevt.query(selden)
    fill_hist(hden, dden[variable])
    fill_hist(hnum, dnum[variable])
    return hden, hnum

def scatterplot(dfevt, nvar1, nvar2, nbins1, min1, max1, nbins2, min2, max2):
    hmult1_mult2 = TH2F(nvar1 + nvar2, nvar1 + nvar2, nbins1, min1, max1, nbins2, min2, max2)
    dfevt_rd = dfevt[[nvar1, nvar2]]
    arr2 = dfevt_rd.values
    fill_hist(hmult1_mult2, arr2)
    return hmult1_mult2

def z_calc(pt_1, phi_1, eta_1, pt_2, phi_2, eta_2):
    np_pt_1 = pt_1.values
    np_pt_2 = pt_2.values
    np_phi_1 = phi_1.values
    np_phi_2 = phi_2.values
    np_eta_1 = eta_1.values
    np_eta_2 = eta_2.values

    cos_phi_1 = np.cos(np_phi_1)
    cos_phi_2 =	np.cos(np_phi_2)
    sin_phi_1 =	np.sin(np_phi_1)
    sin_phi_2 = np.sin(np_phi_2)
    sinh_eta_1 = np.sinh(np_eta_1)
    sinh_eta_2 = np.sinh(np_eta_2)

    px_1 = np_pt_1*cos_phi_1
    px_2 = np_pt_2*cos_phi_2
    py_1 = np_pt_1*sin_phi_1
    py_2 = np_pt_2*sin_phi_2
    pz_1 = np_pt_1*sinh_eta_1
    pz_2 = np_pt_2*sinh_eta_2
    numerator = px_1*px_2+py_1*py_2+pz_1*pz_2
    denominator = px_1*px_1+py_1*py_1+pz_1*pz_1
    return numerator/denominator

def z_gen_calc(pt_1, phi_1, eta_1, pt_2, delta_phi, delta_eta):
    phi_2 = phi_1 + delta_phi
    eta_2 = eta_1 - delta_eta
    return z_calc(pt_1, phi_1, eta_1, pt_2, phi_2, eta_2)

def get_bins(axis):
    return np.array([axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins() + 2)])

def folding(h_input, response_matrix, h_output):
    h_folded=h_output.Clone("h_folded")
    for a in range(h_output.GetNbinsX()):
        for b in range(h_output.GetNbinsY()):
            val=0.0
            val_err=0.0
            for k in range(h_input.GetNbinsX()):
                for l in range(h_input.GetNbinsY()):
                    index_x_out=a+h_output.GetNbinsX()*b
                    index_x_in=k+h_input.GetNbinsX()*l
                    val+=h_input.GetBinContent(k+1,l+1)*response_matrix(index_x_out, index_x_in)
                    val_err+=h_input.GetBinError(k+1,l+1)*h_input.GetBinError(k+1,l+1)*response_matrix(index_x_out, index_x_in)*response_matrix(index_x_out, index_x_in)
            h_folded.SetBinContent(a+1, b+1, val)
            h_folded.SetBinError(a+1, b+1, math.sqrt(val_err))
    return h_folded

def setup_histogram(hist, colour=1, markerstyle=25):
    hist.SetStats(0)
    hist.SetTitleSize(0.04,"X")
    hist.SetTitleOffset(1.0,"X")
    hist.SetTitleSize(0.04,"Y")
    hist.SetTitleOffset(1.0,"Y")
    hist.SetLineWidth(2)
    hist.SetLineColor(colour)
    hist.SetMarkerSize(1.0)
    hist.SetMarkerStyle(markerstyle)
    hist.SetMarkerColor(colour)

def setup_pad(pad):
    pad.SetFillColor(0)
    pad.SetMargin(0.15,0.9,0.15,0.9)
    pad.Draw()
    pad.SetTicks(1,1)
    pad.cd()

def setup_legend(legend):
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    legend.SetTextFont(42)

def setup_tgraph(tg, colour=46, alphastyle=0.3, fillstyle=1001):
    tg.GetXaxis().SetTitleSize(0.04)
    tg.GetXaxis().SetTitleOffset(1.0)
    tg.GetYaxis().SetTitleSize(0.04)
    tg.GetYaxis().SetTitleOffset(1.0)
    tg.SetFillColorAlpha(colour,alphastyle)
    tg.SetLineWidth(2)
    tg.SetLineColor(colour)
    tg.SetFillStyle(fillstyle)
    tg.SetMarkerSize(0)
  
def draw_latex(latex,colour=1,textsize=0.03):
    latex.SetNDC()
    latex.SetTextSize(textsize)
    latex.SetTextColor(colour)
    latex.SetTextFont(42)
    latex.Draw()

def tg_sys(central, variations):
    shapebins_centres=[]
    shapebins_contents=[]
    shapebins_widths_up=[]
    shapebins_widths_down=[]
    shapebins_error_up=[]
    shapebins_error_down=[]

    for i in range(central.GetNbinsX()):
       shapebins_centres.append(central.GetBinCenter(i+1))
       shapebins_contents.append(central.GetBinContent(i+1))
       shapebins_widths_up.append(central.GetBinWidth(i+1)*0.5)
       shapebins_widths_down.append(central.GetBinWidth(i+1)*0.5)
       error_up=0
       error_down=0
       for j in range(len(variations)):
           error = variations[j].GetBinContent(i+1)-central.GetBinContent(i+1)
           if error > 0 and error > error_up :
               error_up = error
           if error < 0 and abs(error) > error_down :
               error_down = abs(error)
       shapebins_error_up.append(error_up)
       shapebins_error_down.append(error_down)
    shapebins_centres_array = array('d',shapebins_centres)
    shapebins_contents_array = array('d',shapebins_contents)
    shapebins_widths_up_array = array('d',shapebins_widths_up)
    shapebins_widths_down_array = array('d',shapebins_widths_down)
    shapebins_error_up_array = array('d',shapebins_error_up)
    shapebins_error_down_array = array('d',shapebins_error_down)
    tg = TGraphAsymmErrors(central.GetNbinsX(),shapebins_centres_array,shapebins_contents_array,shapebins_widths_down_array,shapebins_widths_up_array,shapebins_error_down_array,shapebins_error_up_array)
    return tg
               
           
