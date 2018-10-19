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
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, '../utilities')
from utilitiesRoot import FillNTuple, ReadNTuple, ReadNTupleML
from utilitiesModels import *
from utilitiesPCA import *

time0 = datetime.now()

neventspersample=10000
classifiers, names=getclassifiers()
mylistvariables=getvariablestraining()
mylistvariablesothers=getvariablesothers()
myvariablesy=getvariableissignal()

suffix="SignalN%dBkgN%dPreMassCut" % (neventspersample,neventspersample)
train_set = pd.read_pickle("../buildsample/trainsample%s.pkl" % (suffix))
test_set = pd.read_pickle("../buildsample/testsample%s.pkl" % (suffix))
filenametest_set_ML="output/testsample%sMLdecision.pkl" % (suffix)
ntuplename="fTreeDsFlagged"

myshortlistvariable=["d_len_xy_ML","delta_mass_KK_ML"]
X_train= train_set[myshortlistvariable]
y_train=train_set[myvariablesy]

###################### decision boundaries with 2 most important variables ######################
trainedmodels=fit(names, classifiers,X_train,y_train)
mydecisionboundaries=decisionboundaries(names,trainedmodels,suffix,X_train,y_train)

###################### decision boundaries with 2 first principal components ######################
X_train_2PC,pca=GetPCADataFrame(X_train,myshortlistvariable,2)
trainedmodels=fit(names, classifiers,X_train_2PC,y_train)
mydecisionboundaries=decisionboundaries(names,trainedmodels,suffix+"PCAdecomposition",X_train_2PC,y_train)
