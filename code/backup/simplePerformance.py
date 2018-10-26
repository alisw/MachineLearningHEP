######## code for performing cross validation of the models #########
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
from utilitiesPerformance import *

time0 = datetime.now()

neventspersample=1000
suffix="SignalN%dBkgN%dPreMassCut" % (neventspersample,neventspersample)
ncores=-1

classifiers, names=getclassifiers()
mylistvariables=getvariablestraining()
mylistvariablesothers=getvariablesothers()
myvariablesy=getvariableissignal()

namedataframe="../buildsample/trainsample"+suffix+".pkl"

train_set = pd.read_pickle(namedataframe)
X_train= train_set[mylistvariables]
X_train_others=train_set[mylistvariablesothers]
y_train=train_set[myvariablesy]


#perform score cross validation  
df_scores=cross_validation_mse(names,classifiers,X_train,y_train,10,ncores)
plot_cross_validation_mse(names,df_scores,suffix)

# confusion(mylistvariables,names,classifiers,suffix,X_train,y_train,5)
precision_recall(mylistvariables,names,classifiers,suffix,X_train,y_train,5)
plot_learning_curves(names,classifiers,suffix,X_train,y_train,100,3000,300)
