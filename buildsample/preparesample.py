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
print (type(nevents))

time0 = datetime.now()

mylistvariablespd=['inv_mass_ML*','pt_cand_ML*','d_len_xy_ML*','norm_dl_xy_ML*','cos_p_ML*','cos_p_xy_ML*','imp_par_xy_ML*','sig_vert_ML*',"delta_mass_KK_ML*",'cos_PiDs_ML*',"cos_PiKPhi_3_ML*","signal_ML*"]
mylistvariables = ['inv_mass_ML','pt_cand_ML','d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML","signal_ML"]
myvariablesy='signal_ML'
input_file=("treeTotalSignalN%dBkgN%dPreMassCut.root" % (nevents,nevents))
ntuplename="fTreeDsFlagged"

file = uproot.open(input_file)
tree = file["fTreeDsFlagged"]
dataframeDs=tree.pandas.df(mylistvariablespd)

train_set, test_set = train_test_split(dataframeDs, test_size=0.2, random_state=42)
print("total sample=",len(dataframeDs))
print("train sample=",len(train_set))
print("test sample=",len(test_set))

train_set.to_pickle("../dataframes/trainsamplelSignalN%dBkgN%dPreMassCut.pkl" % (nevents,nevents))
test_set.to_pickle("../dataframes/testsamplelSignalN%dBkgN%dPreMassCut.pkl" % (nevents,nevents))

# train_set_sig=train_set.loc[dataframeDs['signal_ML'] == 1]
# train_set_bkg=train_set.loc[dataframeDs['signal_ML'] == 0]
# 
# figure = plt.figure(figsize=(15,15))
# 
# i=0
# for var in mylistvariables:
#   ax = plt.subplot(4, 3, i+1)  
#   plt.xlabel(var,fontsize=11)
#   plt.ylabel("entries",fontsize=11)
#   plt.yscale('log')
#   kwargs = dict(alpha=0.3,density=True, bins=100)
#   n, bins, patches = plt.hist(train_set_sig[var], facecolor='b', label='signal', **kwargs)
#   n, bins, patches = plt.hist(train_set_bkg[var], facecolor='g', label='background', **kwargs)
#   ax.legend()
#   i=i+1
#   
# plotname='plots/variablesDistribution.png'
# plt.savefig(plotname,bbox_inches='tight')
# 
# 
# mylistvariables1 = ['pt_cand_ML','d_len_xy_ML','sig_vert_ML',"pt_cand_ML","pt_cand_ML","norm_dl_xy_ML","cos_PiDs_ML","cos_p_xy_ML","cos_p_xy_ML"]
# mylistvariables2 = ['d_len_xy_ML','sig_vert_ML','delta_mass_KK_ML',"delta_mass_KK_ML","sig_vert_ML","d_len_xy_ML","cos_PiKPhi_3_ML","sig_vert_ML","pt_cand_ML"]
# 
# figurecorr = plt.figure(figsize=(30,20))
# i=0
# for i in range(len(mylistvariables1)):
#   axcorr = plt.subplot(3, 3, i+1) 
#   plt.xlabel(mylistvariables1[i],fontsize=11)
#   plt.ylabel(mylistvariables2[i],fontsize=11)
#   plt.scatter(train_set_bkg[mylistvariables1[i]], train_set_bkg[mylistvariables2[i]], alpha=0.4, c="g",label="background")
#   plt.scatter(train_set_sig[mylistvariables1[i]], train_set_sig[mylistvariables2[i]], alpha=0.4, c="b",label="signal")
#   plt.title('Pearson sgn: %s'%train_set_sig.corr().loc[mylistvariables1[i]][mylistvariables2[i]].round(2)+',  Pearson bkg: %s'%train_set_bkg.corr().loc[mylistvariables1[i]][mylistvariables2[i]].round(2))
#   axcorr.legend()
#   i=i+1
# 
# plotname='plots/variablesCorrelation.png'
# plt.savefig(plotname,bbox_inches='tight')
# 
# 
# f, ax = plt.subplots(figsize=(10, 8))
# corr = train_set_sig.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)
# 
# 
# plotname='plots/variablesscattermatrix.png'
# plt.savefig(plotname,bbox_inches='tight')
# 
# ### trying some principal component analysis, but there is no strong correlations among the feaures 
# # so it should not be needed
# pca = PCA(n_components=5)
# pca.fit(train_set_sig)
# pca_score = pca.explained_variance_ratio_
# V = pca.components_
# print (V)