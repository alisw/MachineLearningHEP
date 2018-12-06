###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##     Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to: study selection efficiency and expected significance
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ROOT import TH1F, gROOT
from DataBaseMLparameters import getmasscut, getOptimizationParameters

def calc_efficiency(df_to_sel, name, num_step):
  x_axis = np.linspace(0, 1.00, num_step)
  num_tot_cand = len(df_to_sel)
  eff_array = []
  
  for thr in x_axis:
    num_sel_cand = len(df_to_sel[ df_to_sel['y_test_prob' + name].values >= thr])
    eff_array.append(num_sel_cand/num_tot_cand)

  return eff_array, x_axis   

def calc_bkg(df_bkg, name, num_step, mass_cuts, fit_region, bin_width, sig_region):
  x_axis = np.linspace(0, 1.00, num_step)
  bkg_array = []
  num_bins = (fit_region[1] - fit_region[0]) / bin_width
  num_bins = int(round(num_bins))
  bin_width = (fit_region[1] - fit_region[0]) / num_bins
  bkg_mass_mask = (df_bkg['inv_mass_ML'].values <= mass_cuts[0]) | (df_bkg['inv_mass_ML'].values >= mass_cuts[1])
  df_mass = df_bkg[bkg_mass_mask]
  
  for thr in x_axis:
    bkg = 0.
    hmass = TH1F('hmass', '', num_bins, fit_region[0], fit_region[1])
    bkg_sel_mask = df_mass['y_test_prob' + name].values >= thr
    sel_mass_array = df_mass[bkg_sel_mask]['inv_mass_ML'].values

    if len(sel_mass_array) > 5 : 
      for mass_value in np.nditer(sel_mass_array):
        hmass.Fill(mass_value)

      fit = hmass.Fit('expo', 'Q', '',  fit_region[0], fit_region[1])  
      if(int(fit) == 0):
        fit_func = hmass.GetFunction('expo')
        bkg = fit_func.Integral(sig_region[0], sig_region[1]) / bin_width
        del fit_func

    bkg_array.append(bkg)
    del hmass
    
  return bkg_array, x_axis

def calc_signif(sig_array, bkg_array):
  signif_array = []

  for sig, bkg in zip(sig_array, bkg_array):
    signif = 0
    if sig > 0 and bkg > 0:    
      signif = sig/np.sqrt(sig + bkg)
    signif_array.append(signif)

  return signif_array    

def plot_FONLL(common_dict, part_label, suffix, plot_dir):
  df = pd.read_csv(common_dict['filename'])
  figure = plt.figure(figsize=(20,15))
  ax=plt.subplot(111)
  plt.plot(df['pt'], df['max'] * common_dict['FF'], linewidth=4.0)
  plt.xlabel('P_t [GeV/c]', fontsize=20)
  plt.ylabel('Cross Section [pb/GeV]', fontsize=20)
  plt.title("FONLL cross section " + part_label, fontsize=20)  
  plt.semilogy()
  plot_name=plot_dir + '/FONLL curve %s.png' % (suffix)
  plt.savefig(plot_name)

  return

def calc_sig_Dmeson(common_dict, pt_specific_dict, pt_min, pt_max):
  df = pd.read_csv(common_dict['filename'])
  df_in_pt = df.query('(pt >= @pt_min) and (pt < @pt_max)')['max']
  prod_cross = df_in_pt.sum() * common_dict['FF'] * 1e-12 / len(df_in_pt)
  delta_pt = pt_max - pt_min
  sig_before_sel = 2. * prod_cross * delta_pt * common_dict['BR'] * pt_specific_dict['acc_times_pre_sel'] * common_dict['n_events'] / (common_dict['sigma_MB'] * pt_specific_dict['f_prompt'])

  return sig_before_sel 

def studysignificance(part_label, pt_min, pt_max, sig_set, bkg_set, names, target_var, suffix, plot_dir):
  gROOT.SetBatch(True)
  gROOT.ProcessLine("gErrorIgnoreLevel = 2000;")

  common_dict, pt_specific_dict = getOptimizationParameters(part_label, pt_min, pt_max)
  plot_FONLL(common_dict, part_label, suffix, plot_dir)
  sig_before_sel = calc_sig_Dmeson(common_dict, pt_specific_dict, pt_min, pt_max)  
  mass = common_dict['mass']
  sigma = pt_specific_dict['sigma']
  sig_region = [mass - 3 * sigma, mass + 3 * sigma]
  mass_min, mass_max = getmasscut(part_label)  
  mass_cuts = [mass_min, mass_max]
  
  fig_eff = plt.figure(figsize = (20,15))
  plt.xlabel('Probability', fontsize=20)
  plt.ylabel('Efficiency', fontsize=20)
  plt.title("Efficiency Signal", fontsize=20)

  fig_signif = plt.figure(figsize = (20,15))
  plt.xlabel('Probability', fontsize=20)
  plt.ylabel('Significance (A.U.)', fontsize=20)
  plt.title("Significance vs probability ",fontsize=20)

  num_steps = 101
  
  for name in names:
    mask_sig = sig_set[target_var].values == 1
    eff_array, x_axis = calc_efficiency(sig_set[mask_sig], name, num_steps)
    plt.figure(fig_eff.number)
    plt.plot(x_axis, eff_array, alpha = 0.3, label = '%s' % name, linewidth = 4.0)

    sig_array = [ eff * sig_before_sel for eff in eff_array]
    mask_bkg = bkg_set['cand_type_ML'].values == 0
    bkg_array, x_axis_bkg = calc_bkg(bkg_set[mask_bkg], name, num_steps, mass_cuts, common_dict['mass_fit_lim'], pt_specific_dict['bin_width'], sig_region)
    
    signif_array = calc_signif(sig_array, bkg_array)
    plt.figure(fig_signif.number)
    plt.plot(x_axis, signif_array, alpha = 0.3, label = '%s' % name, linewidth = 4.0)

  plt.figure(fig_eff.number)
  plt.legend(loc="lower left", prop={'size':18})
  plt.savefig(plot_dir + '/Efficiency%sSignal.png' % suffix)

  plt.figure(fig_signif.number)
  plt.legend(loc="lower left",  prop={'size':18})
  plt.savefig(plot_dir + '/Significance%s.png' % suffix)
  
  return