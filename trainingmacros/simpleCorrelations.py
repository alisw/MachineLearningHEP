import array
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import sys, os
from timeit import default_timer as timer
from datetime import datetime
from ROOT import TNtuple
from ROOT import TH1F, TH2F, TCanvas, TFile, gStyle, gROOT
import uproot

import sys
sys.path.insert(0, '../utilities')
from utilitiesRoot import FillNTuple, ReadNTuple, ReadNTupleML
from utilitiesModels import *
from utilitiesGeneral import *
from utilitiesCorrelations import *

time0 = datetime.now()

neventspersample=10000
ptmin=5
ptmax=7
var_pt="pt_cand_ML"
var_signal="signal_ML"
path = "./plots/%.1f_%.1f_GeV"%(ptmin,ptmax)
checkdir(path)

classifiers, names=getclassifiers()
mylistvariables=getvariablestraining()
mylistvariablesothers=getvariablesothers()
myvariablesy=getvariableissignal()

suffix="SignalN%dBkgN%dPreMassCut" % (neventspersample,neventspersample)
train_set = pd.read_pickle("../buildsample/trainsample%s.pkl" % (suffix))

print (train_set.pt_cand_ML)
X_train= train_set[mylistvariables]
y_train=train_set[myvariablesy]

################ training set ##################
train_set_ptsel=filterdataframe_pt(train_set,var_pt,ptmin,ptmax)
train_set_ptsel_sig,train_set_ptsel_bkg=splitdataframe_sigbkg(train_set_ptsel,var_signal)
######## variable distribution plots ###########
vardistplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariables,path)
######## variable scatter plots ###########
# mylistvariablesx = ['pt_cand_ML','d_len_xy_ML','sig_vert_ML',"pt_cand_ML","pt_cand_ML","norm_dl_xy_ML","cos_PiDs_ML","cos_p_xy_ML","cos_p_xy_ML"]
# mylistvariablesy = ['d_len_xy_ML','sig_vert_ML','delta_mass_KK_ML',"delta_mass_KK_ML","sig_vert_ML","d_len_xy_ML","cos_PiKPhi_3_ML","sig_vert_ML","pt_cand_ML"]
# n,bins, patches = plt.hist(train_set_ptsel.pt_cand_ML,50)
# plt.show()
