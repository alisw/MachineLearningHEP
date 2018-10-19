import array
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import pickle
import sys, os
from timeit import default_timer as timer
from datetime import datetime


def filterdataframe_pt(dataframe_,pt_var_,ptmin_,ptmax_):
  dataframe_ptsel_=dataframe_.loc[(dataframe_[pt_var_] > ptmin_ ) & (dataframe_[pt_var_] < ptmax_ )]
  return dataframe_ptsel_
  
def splitdataframe_sigbkg(dataframe_,var_signal_):
  dataframe_sig_=dataframe_.loc[dataframe_[var_signal_] == 1]
  dataframe_bkg_=dataframe_.loc[dataframe_[var_signal_] == 0]
  return dataframe_sig_,dataframe_bkg_

def checkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)
    
def progressbar(part,tot):
  sys.stdout.flush()      
  length=100
  perc = part/tot
  num_dashes = int(length*perc)
  print("\r[",end='')
  for i in range(0,num_dashes+1):
    print("#",end='')
  for i in range(0,length-num_dashes-1):
    print("-",end='')
  print("] {0:.0%}".format(perc),end='')

