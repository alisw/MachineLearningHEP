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

time0 = datetime.now()

neventspersample=100000
classifiers, names=getclassifiers()
mylistvariables=getvariablestraining()
mylistvariablesothers=getvariablesothers()
myvariablesy=getvariableissignal()

suffix="SignalN%dBkgN%dPreMassCut" % (neventspersample,neventspersample)
train_set = pd.read_pickle("../buildsample/trainsample%s.pkl" % (suffix))
test_set = pd.read_pickle("../buildsample/testsample%s.pkl" % (suffix))
filenametest_set_ML="output/testsample%sMLdecision.pkl" % (suffix)
ntuplename="fTreeDsFlagged"

X_train= train_set[mylistvariables]
X_train_others=train_set[mylistvariablesothers]
y_train=train_set[myvariablesy]

X_test= test_set[mylistvariables]
X_test_others=test_set[mylistvariablesothers]
y_test=test_set[myvariablesy]

###################### training sequence ######################
trainedmodels=fit(names, classifiers,X_train,y_train)
print ('Training time')
print (datetime.now() - time0)

###################### importance study ######################
importanceplotall(mylistvariables,names,trainedmodels,suffix)

###################### saving model ######################
savemodels(names,trainedmodels,"output",suffix)
###################### testing sequence ######################
time1 = datetime.now()
test_setML=test(names,trainedmodels,X_test,test_set)
test_set.to_pickle(filenametest_set_ML)

print ('Testing time')
print (datetime.now() - time1)
