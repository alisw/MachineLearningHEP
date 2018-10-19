import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import seaborn as sns

def vardistplot(dataframe_sig_,dataframe_bkg_,mylistvariables_,output_):
  figure = plt.figure(figsize=(20,15))
  i=1
  for var in mylistvariables_:
    ax = plt.subplot(len(mylistvariables_)/4+1, len(mylistvariables_)/3, i)  
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


def scatterplot(dataframe_sig_,dataframe_bkg_,mylistvariablesx_,mylistvariablesy_,output_):
  figurecorr = plt.figure(figsize=(30,20))
  i=1
  for i in range(len(mylistvariablesx_)):
    axcorr = plt.subplot(3, 3, i+1) 
    plt.xlabel(mylistvariablesx_[i],fontsize=11)
    plt.ylabel(mylistvariablesy_[i],fontsize=11)
    plt.scatter(dataframe_bkg_[mylistvariablesx_[i]], dataframe_bkg_[mylistvariablesy_[i]], alpha=0.4, c="g",label="background")
    plt.scatter(dataframe_sig_[mylistvariablesx_[i]], dataframe_sig_[mylistvariablesy_[i]], alpha=0.4, c="b",label="signal")
    plt.title('Pearson sgn: %s'%dataframe_sig_.corr().loc[mylistvariablesx_[i]][mylistvariablesy_[i]].round(2)+',  Pearson bkg: %s'%dataframe_bkg_.corr().loc[mylistvariablesx_[i]][mylistvariablesy_[i]].round(2))
    axcorr.legend()
    i=i+1
  plotname=output_+'/variablesScatterPlot.png'
  plt.savefig(plotname,bbox_inches='tight')
   
def correlationmatrix(dataframe,output_,label):
  corr = dataframe.corr()
  f, ax = plt.subplots(figsize=(10, 8))
  plt.title(label,fontsize=11)
  sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
  plotname=output_+'/correlationmatrix'+label+'.png'
  plt.savefig(plotname,bbox_inches='tight')

