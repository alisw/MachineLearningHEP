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
mylistvariablesall=getvariablesall()
mylistvariablesothers=getvariablesothers()
myvariablesy=getvariableissignal()
mylistvariablescorrx,mylistvariablescorry=getvariablecorrelation()

suffix="SignalN%dBkgN%dPreMassCut" % (neventspersample,neventspersample)
train_set = pd.read_pickle("../buildsample/trainsample%s.pkl" % (suffix))

print (train_set.pt_cand_ML)
X_train= train_set[mylistvariables]
y_train=train_set[myvariablesy]

################ creating signal and bkg dataset ##################
train_set_ptsel=filterdataframe_pt(train_set,var_pt,ptmin,ptmax)
train_set_ptsel_sig,train_set_ptsel_bkg=splitdataframe_sigbkg(train_set_ptsel,var_signal)

######## single variable distribution plots ###########
vardistplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariablesall,path)

######## variable scatter plots ###########
mylistvariablesx,mylistvariablesy=getvariablecorrelation()
scatterplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariablesx,mylistvariablesy,path)

######## correlation matrix #################
# correlationmatrix(train_set_sig,train_set_bkg,path)

time1 = datetime.now()
howmuchtime = time1-time0
print("\n===\n===\tExecution END. Start time: %s\tEnd time: %s\t(%s)\n===\n\n\n"%(time0.strftime('%d/%m/%Y, %H:%M:%S'),time1.strftime('%d/%m/%Y, %H:%M:%S'),howmuchtime))
