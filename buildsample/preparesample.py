import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import array
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import uproot
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import sys, os
from timeit import default_timer as timer
from datetime import datetime
from ROOT import TNtuple
from ROOT import TH1F, TH2F, TCanvas, TFile, gStyle, gROOT
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns


import sys
print (type(sys.argv[1]))
nevents=int(sys.argv[1])
sys.path.insert(0, '../utilities')

from BinaryMultiFeaturesClassification import getvariablestraining,getvariablesothers,getvariableissignal,getvariablesall,getvariablecorrelation
from utilitiesGeneral import preparestringforuproot
print (type(nevents))

time0 = datetime.now()
case="Ds"


mylistvariables=getvariablestraining(case)
mylistvariablesothers=getvariablesothers(case)
myvariablesy=getvariableissignal(case)

input_file=("treeTotalSignalN%dBkgN%dPreMassCut.root" % (nevents,nevents))
ntuplename="fTreeDsFlagged"

file = uproot.open(input_file)
tree = file["fTreeDsFlagged"]
totarrayvariables=mylistvariables+mylistvariablesothers+[myvariablesy]
print (totarrayvariables)
dataframeDs=tree.pandas.df(preparestringforuproot(totarrayvariables))

train_set, test_set = train_test_split(dataframeDs, test_size=0.2, random_state=42)
print("total sample=",len(dataframeDs))
print("train sample=",len(train_set))
print("test sample=",len(test_set))

train_set.to_pickle("trainsampleSignalN%dBkgN%dPreMassCut%s.pkl" % (nevents,nevents,case))
test_set.to_pickle("testsampleSignalN%dBkgN%dPreMassCut%s.pkl" % (nevents,nevents,case))
