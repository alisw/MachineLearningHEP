import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

def vardistplot(dataframe_sig_,dataframe_bkg_,mylistvariables_,output_):
  figure = plt.figure(figsize=(15,15))
  i=1
  for var in mylistvariables_:
    ax = plt.subplot(len(mylistvariables_)/3+1, len(mylistvariables_)/3, i)  
    plt.xlabel(var,fontsize=11)
    plt.ylabel("entries",fontsize=11)
    plt.yscale('log')
    kwargs = dict(alpha=0.3,density=True, bins=100)
    n, bins, patches = plt.hist(dataframe_sig_[var], facecolor='b', label='signal', **kwargs)
    n, bins, patches = plt.hist(dataframe_bkg_[var], facecolor='g', label='background', **kwargs)
    ax.legend()
    i=i+1   
  plotname=output_+'/variablesDistribution.png'
  plt.savefig(plotname,bbox_inches='tight')


# def vardistplot(dataframe_sig_,dataframe_bkg_,mylistvariables_,output_):
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
