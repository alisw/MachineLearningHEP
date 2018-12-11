###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to: apply Principal Component Analysis (PCA) and to standardize features
"""

import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def plotvariancePCA(PCA_object,output_):
  figure = plt.figure(figsize=(15,10))
  plt.plot(np.cumsum(PCA_object.explained_variance_ratio_))
  plt.plot([0,10],[0.95,0.95])
  plt.xlabel('number of components',fontsize=16)
  plt.ylabel('cumulative explained variance',fontsize=16)
  plt.title('Explained variance',fontsize=16)
  plt.ylim([0, 1])
  plotname=output_+'/PCAvariance.png'
  plt.savefig(plotname,bbox_inches='tight')

# def createinfoplot(array_sig,array_bkg,path):
#   position = np.arange(1,len(array_sig)+1,dtype = int)
#   list_info_sig = []
#   list_info_bkg = []
#   info_sig = 0
#   info_bkg = 0
#   for i in np.arange(len(position)):
#     info_sig += array_sig[i]
#     info_bkg += array_bkg[i]
#     list_info_sig.append(info_sig)
#     list_info_bkg.append(info_bkg)
#   arr_info_sig = np.array(list_info_sig)
#   arr_info_bkg = np.array(list_info_bkg)
#   #print(position)
#   #print(arr_info_sig)
#   #print(arr_info_bkg)   
#   infoplot = plt.figure(figsize=(20,20))
#   infopad = plt.subplot(1,1,1)
#   plt.plot(position,arr_info_sig,'-ro',markersize=15)        
#   plt.plot(position,arr_info_bkg,'-bo',markersize=15)
#   infopad.set_xlabel('number of principal components',fontsize=25)        
#   infopad.set_ylabel('total carried information',fontsize=25)
#   infopad.xaxis.set_tick_params(labelsize=25)
#   infopad.yaxis.set_tick_params(labelsize=25)
#   plt.rc('ytick',labelsize=25)
#   infopad.legend(("signal","background"),fontsize=50)
#   plt.grid()
#   plt.savefig(path+"/../carried_info.pdf")
