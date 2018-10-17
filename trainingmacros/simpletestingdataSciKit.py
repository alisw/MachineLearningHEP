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

classifiers, names=getclassifiers()
mylistvariables=getvariablestraining()
mylistvariablesothers=getvariablesothers()

bigntupla="/Users/gianmicheleinnocenti/MLDsproductions/Data/2018Sep21_LHC15o_pass1_pidfix/AnalysisResults_000_root6.root"

neventspersample=10000
suffix="SignalN%dBkgN%dPreMassCut.pkl" % (neventspersample,neventspersample)
bigntuplaML="output/AnalysisResults_000_%s_MLdecision.root" % (suffix)

filedata = uproot.open(bigntupla)
treedata = filedata["fTreeDsData"]

frame_X_test=treedata.pandas.df(preparestringforuproot(mylistvariables))
frame_X_test_others=treedata.pandas.df(preparestringforuproot(mylistvariablesothers))

X_test= frame_X_test[mylistvariables]
X_test_others=frame_X_test_others[mylistvariablesothers]

trainedmodels=readmodels(names,"output",suffix)

X_test_all=X_test
for name, model in zip(names, trainedmodels):
  y_test_prediction=[]
  y_test_prob=[]
  y_test_prediction=model.predict(X_test)
  y_test_prob=model.predict_proba(X_test)[:,1]
  X_test_all = np.c_[X_test_all,y_test_prediction]
  mylistvariables.append('y_test_prediction'+name)
  X_test_all = np.c_[X_test_all,y_test_prob]
  mylistvariables.append('y_test_prob'+name)

mylistvariables=mylistvariables+mylistvariablesothers

X_test_all = np.c_[X_test_all,X_test_others]
f_out = TFile.Open(bigntuplaML,"recreate")
FillNTuple("fTreeDsData",X_test_all,mylistvariables)
f_out.Close()

