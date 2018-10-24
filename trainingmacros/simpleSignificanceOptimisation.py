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
from utilitiesOptimisation import *

time0 = datetime.now()

neventspersample=100000
ptmin=5
ptmax=7

classifiers, names=getclassifiers()
mylistvariables=getvariablestraining()
mylistvariablesothers=getvariablesothers()
myvariablesy=getvariableissignal()

suffix="SignalN%dBkgN%dPreMassCut" % (neventspersample,neventspersample)
filenametest_set_ML="output/testsample%sMLdecision.pkl" % (suffix)
test_set_ML = pd.read_pickle(filenametest_set_ML)

df= pd.read_csv('../../fonll/fo_pp_d0meson_5TeV_y0p5.csv')
plotfonll(df.pt,df.central,"D0")
signalfonll=getfonll(df,ptmin,ptmax)

sig=100
bkg=10000

efficiencySig_array,xaxisSig,num_arraySig,den_arraySig=get_efficiency_effnum_effden(test_set_ML,names,"signal_ML",1,0.01)
efficiencyBkg_array,xaxisBkg,num_arrayBkg,den_arrayBkg=get_efficiency_effnum_effden(test_set_ML,names,"signal_ML",0,0.01)

plot_efficiency(names,efficiencySig_array,xaxisSig,"signal",suffix)
plot_efficiency(names,efficiencyBkg_array,xaxisBkg,"background",suffix)

significance_array, xaxis= calculatesignificance(efficiencySig_array,sig,efficiencyBkg_array,bkg,xaxisSig)
plot_significance(names,significance_array,xaxis,suffix)


