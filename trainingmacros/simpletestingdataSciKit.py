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
bigntuplaML="output/AnalysisResults_000_MLdecision.root"

filedata = uproot.open(bigntupla)
treedata = filedata["fTreeDsData"]

frame_X_test=treedata.pandas.df(preparestringforuproot(mylistvariables))
frame_X_test_others=treedata.pandas.df(preparestringforuproot(mylistvariablesothers))

X_test= frame_X_test[mylistvariables]
X_test_others=frame_X_test_others[mylistvariablesothers]

X_test_all=X_test
for name, clf in zip(names, classifiers):
  y_test_prediction=[]
  y_test_prob=[]
  fileoutmodel = "models/"+name+".sav"
  model = pickle.load(open(fileoutmodel, 'rb'))
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

