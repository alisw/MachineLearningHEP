import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getfonllintegrated(df_,ptmin_,ptmax_):
  sumcross=0
  for i in df_.pt:
    if (i>=ptmin_ and i<ptmax_):
      sumcross=sumcross+i
  return sumcross


def get_efficiency_effnum_effden(df_,names_,selvar_,flag_label,stepsize):
  xaxis_=np.arange(start=0,stop=1.00,step=stepsize)
  df_sel=df_.loc[df_[selvar_] == flag_label]
  efficiency_array=[]
  num_array=[]
  den_array=[]
  x_array=[]
  for name in names_:
    deneff=np.full((len(xaxis_),1),len(df_sel))
    numeff=np.full((len(xaxis_),1),0)
    probability=df_sel['y_test_prob'+name]
    for t,threshold in enumerate(xaxis_):
      for prob in probability:
        if (prob>=threshold):
          numeff[t]=numeff[t]+1
    seleff=numeff/deneff
    x_array.append(xaxis_)
    num_array.append(numeff)
    den_array.append(deneff)
    efficiency_array.append(seleff)
  return efficiency_array,x_array,num_array,den_array
    

def plot_efficiency(names_,efficiency_array,xaxis_,label,suffix_):

  figure = plt.figure(figsize=(20,15))
  i=1
  for name in names_:
    plt.xlabel('Probability',fontsize=20)
    plt.ylabel('Efficiency',fontsize=20)
    plt.title("Efficiency "+label,fontsize=20)
    plt.plot(xaxis_[i-1], efficiency_array[i-1], lw=1, alpha=0.3, label='%s' % (names_[i-1]), linewidth=4.0)
    plt.legend(loc="lower center",  prop={'size':18})
    i += 1
  plotname='plots/efficiency%s%s.png' % (label,suffix_)
  plt.savefig(plotname)
  
def calculatesignificance(efficiencySig_array,sig, efficiencyBkg_array, bkg):
  significance_array=[]
  for i,name in enumerate(efficiencySig_array):
    signal=efficiencySig_array[i]*sig;
    bkg=efficiencyBkg_array[i]*bkg;
    significance=signal/np.sqrt(signal+bkg)
    significance_array.append(significance)
  return significance_array
    
def plot_significance(names_,significance_array,xaxis_,suffix,plotdir):

  figure = plt.figure(figsize=(20,15))
  i=1
  for name in names_:
    plt.xlabel('Probability',fontsize=20)
    plt.ylabel('Significance (A.U.)',fontsize=20)
    plt.title("Significance vs probability ",fontsize=20)
    plt.plot(xaxis_[i-1], significance_array[i-1], lw=1, alpha=0.3, label='%s' % (names_[i-1]), linewidth=4.0)
    plt.legend(loc="lower center",  prop={'size':18})
    i += 1
  plotname=plotdir+'/Significance%s.png' % (suffix)
  plt.savefig(plotname)

def plotfonll(pt_array,cross_array,particlelabel,suffix,plotdir):
  figure = plt.figure(figsize=(20,15))
  ax=plt.subplot(111)
  plt.xlabel('pt',fontsize=20)
  plt.ylabel('cross section',fontsize=20)
  plt.title("FONLL cross section "+particlelabel,fontsize=20)
  plt.plot(pt_array,cross_array,linewidth=4.0)
  plt.semilogy()
  plotname=plotdir+'/FONLL curve %s.png' % (suffix)
  plt.savefig(plotname)
