import array
import numpy as np
import pandas as pd
import math
import matplotlib

import pickle
import sys, os
from timeit import default_timer as timer
from datetime import datetime
from util import init,fit,cross_validation_mse,plot_cross_validation_mse,studyMLalgorithm,do_gridsearch,importanceplotall,plot_gridsearch


dofit=0
doimportanceplotall=0
docross=0
doplotcross=0
dogridsearch=1
doplotgridsearch=1

nevents=10000
ncores=-1

time0 = datetime.now()
classifiers,names,mylistvariables,mylistvariablesothers,myvariablesy=init()

train_set = pd.read_pickle("dataframes/trainsample%d.pkl" % (nevents))

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

X_train= train_set[mylistvariables]
y_train=train_set[myvariablesy]


if (dofit==1): 
  fit(names, classifiers,X_train,y_train)
  print ('fit done')
if (doimportanceplotall==1): 
  importanceplotall(mylistvariables,names)
  print ('importance done')
if (docross==1): 
  df_scores=cross_validation_mse(names,classifiers,X_train,y_train,10,ncores)
  print ('cross done')
if (doplotcross==1): 
  plot_cross_validation_mse(names)
  print ('plot cross done')
if (dogridsearch==1): 
  do_gridsearch(mylistvariables,X_train,y_train,5,ncores)
  print ('grid search done')
if (doplotgridsearch==1): 
  plot_gridsearch()
  print ('plot grid search done')

print ('Training time')
print (datetime.now() - time0)
time1 = datetime.now()
