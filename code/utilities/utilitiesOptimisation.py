###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to: study selection efficiency and expected significance
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataBaseMLparameters import getmasscut
from ROOT import TH1F, gROOT

def calc_efficiency(df_, name_, target_var_, flag_label_, num_step_):
  x_axis_ = np.linspace(0, 1.00, num_step_)
  df_to_sel = df_[ df_[target_var_].values == flag_label_]
  num_tot_cand = len(df_to_sel)
  eff_array = []
  
  for thr in x_axis_:
    num_sel_cand = len(df_to_sel[ df_to_sel['y_test_prob' + name_].values >= thr])
    eff_array.append(num_sel_cand/num_tot_cand)

  return eff_array, x_axis_    

def calc_bkg(df_, name_, target_var_, flag_label_, num_step_, mass_min_cut_, mass_max_cut_, sig_region_):
  x_axis_ = np.linspace(0, 1.00, num_step_)
  df_bkg = df_[ df_[target_var_].values == 0]
  bkg_array = []

  num_bins = 40
  bin_range = [1.75, 2.15]
  bin_width = (bin_range[1] - bin_range[0]) / num_bins
  
  for thr in x_axis_:
    bkg = 0.
    hmass = TH1F('hmass', '', num_bins, bin_range[0], bin_range[1])

    for index, row in df_bkg.iterrows():
      if ( (row['inv_mass_ML'] <= mass_min_cut_ or row['inv_mass_ML'] >= mass_max_cut_) and row['y_test_prob' + name_] >= thr ):
        hmass.Fill(row['inv_mass_ML'])

    if(hmass.GetEntries() > 10):
      fit = hmass.Fit('expo', 'Q', '',  bin_range[0], bin_range[1])  
      if(int(fit) == 0):
        fit_func = hmass.GetFunction('expo')
        bkg = fit_func.Integral(sig_region_[0], sig_region_[1]) / bin_width
        del fit_func

    bkg_array.append(bkg)
    del hmass
    
  return bkg_array, x_axis_

def calc_signif(sig_array_, bkg_array_):
  signif_array = []

  for sig, bkg in zip(sig_array_, bkg_array_):
    signif = 0
    if sig > 0 and bkg > 0:    
      signif = sig/np.sqrt(sig + bkg)
    signif_array.append(signif)

  return signif_array    

def plotfonll(pt_array,cross_array,particlelabel,suffix,plotdir):
  figure = plt.figure(figsize=(20,15))
  ax=plt.subplot(111)
  plt.xlabel('P_t [GeV/c]',fontsize=20)
  plt.ylabel('Cross Section [pb/GeV]',fontsize=20)
  plt.title("FONLL cross section "+particlelabel,fontsize=20)
  plt.plot(pt_array,cross_array,linewidth=3.0)
  plt.semilogy()
  plotname=plotdir+'/FONLL curve %s.png' % (suffix)
  plt.savefig(plotname)

  return

def getFONLLdataframe_FF(case):
  filename=""
  FF=-1.
  if (case=="Ds"):
    filename='../fonll/fo_pp_d0meson_5TeV_y0p5.csv'
    FF= 0.61 * 0.21
  if (case=="Dplus"):
    filename=''
    FF=0.
  if (case=="Lc"):
    filename=""
    FF=0.
  if (case=="Bplus"):
    filename==""
    FF=0.
  df= pd.read_csv(filename)
  
  return df,FF

def studysignificance(optionAnalysis,ptmin,ptmax,test_set,names,target_var,suffix,plotdir):

  gROOT.SetBatch(True)
  gROOT.ProcessLine("gErrorIgnoreLevel = 2000;")

  df,FF= getFONLLdataframe_FF(optionAnalysis)
  plotfonll(df.pt,df.central*FF,optionAnalysis,suffix,plotdir)

  n_events = 9.9586758e+08 
  sigma_MB = 51.2 * 1e-3
  BR = 2.27 * 1e-2   
  f_prompt = 0.9
  acc_times_pre_sel = 1.3 * 0.3
  mass_Ds = 1.972
  sigma_Ds = 0.009
  sig_region = [mass_Ds - 3 * sigma_Ds, mass_Ds + 3 * sigma_Ds]

  df_in_pt = df.query('(pt >= @ptmin) and (pt < @ptmax)')['max']

  prod_cross = df_in_pt.sum() * FF * 1e-12 / len(df_in_pt)
  delta_pt = ptmax - ptmin
  signal_before_sel = 2. * prod_cross * delta_pt * BR * acc_times_pre_sel * n_events / (sigma_MB * f_prompt)
  mass_min, mass_max = getmasscut(optionAnalysis)  
  
  fig_eff = plt.figure(figsize = (20,15))
  plt.xlabel('Probability', fontsize=20)
  plt.ylabel('Efficiency', fontsize=20)
  plt.title("Efficiency Signal", fontsize=20)

  fig_signif = plt.figure(figsize = (20,15))
  plt.xlabel('Probability', fontsize=20)
  plt.ylabel('Significance (A.U.)', fontsize=20)
  plt.title("Significance vs probability ",fontsize=20)
  
  for name in names:
    eff_array, x_axis = calc_efficiency(test_set, name, target_var, 1, 101)
    plt.figure(fig_eff.number)
    plt.plot(x_axis, eff_array, alpha = 0.3, label = '%s' % name, linewidth = 4.0)

    sig_array = [ eff * signal_before_sel for eff in eff_array]
    bkg_array, x_axis_bkg = calc_bkg(test_set, name, target_var, 0, 101, mass_min, mass_max, sig_region)
    bkg_array = [ bkg * 400 for bkg in bkg_array]

    signif_array = calc_signif(sig_array, bkg_array)
    plt.figure(fig_signif.number)
    plt.plot(x_axis, signif_array, alpha = 0.3, label = '%s' % name, linewidth = 4.0)

  plt.figure(fig_eff.number)
  plt.legend(loc="lower left", prop={'size':18})
  plt.savefig(plotdir + '/Efficiency%sSignal.png' % suffix)

  plt.figure(fig_signif.number)
  plt.legend(loc="lower left",  prop={'size':18})
  plt.savefig(plotdir + '/Significance%s.png' % suffix)
  
  return