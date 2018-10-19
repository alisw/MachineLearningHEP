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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def GetPCADataFrameAndPC(dataframe,n_pca):
  data_values = dataframe.values
  pca = PCA(n_pca)
  principalComponent = pca.fit_transform(data_values)
  pca_name_list = []
  for i_pca in range(n_pca):
    pca_name_list.append("princ_comp_%d"%(i_pca+1))
  pca_dataframe = pd.DataFrame(data=principalComponent,columns=pca_name_list)
  return pca_dataframe, pca


def GetDataFrameStandardised(dataframe):
  listheaders=list(dataframe.columns.values)
  data_values = dataframe.values
  data_values_std = StandardScaler().fit_transform(data_values)
  dataframe_std = pd.DataFrame(data=data_values_std,columns=listheaders)
  return dataframe_std
