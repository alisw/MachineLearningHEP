from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from utilitiesGeneral import filterdataframe_pt,splitdataframe_sigbkg,checkdir,preparestringforuproot
import pandas as pd
import numpy as np
import uproot


def getvariablestraining(case):
  if (case=="Ds"):
    mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"]
  if (case=="Lc"):
    mylistvariables=['d_len_ML','d_len_xy_ML','norm_dl_xy_ML','dist_12_ML','cos_p_ML','pt_p_ML','pt_K_ML','pt_pi_ML','sig_vert_ML','dca_ML']
  return mylistvariables

def getvariablesothers(case):
  if (case=="Ds" or case=="Lc"):
    mylistvariablesothers=['inv_mass_ML','pt_cand_ML']
  return mylistvariablesothers

def getvariableissignal(case):
  if (case=="Ds" or case=="Lc"):
    myvariablesy='signal_ML'
  return myvariablesy

def getvariablesall(case):
  if (case=="Ds"):
    mylistvariablesall=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML",'inv_mass_ML','pt_cand_ML','signal_ML',"cand_type_ML"]
  if (case=="Lc"):
    mylistvariablesall=['inv_mass_ML','pt_cand_ML','d_len_ML','d_len_xy_ML','norm_dl_xy_ML','dist_12_ML','cos_p_ML','pt_p_ML','pt_K_ML','pt_pi_ML','sig_vert_ML','dca_ML','cand_type_ML']
  return mylistvariablesall

def getvariablecorrelation(case):
  if (case=="Ds"):
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML','sig_vert_ML',"pt_cand_ML","pt_cand_ML","norm_dl_xy_ML","cos_PiDs_ML","cos_p_xy_ML","cos_p_xy_ML"]
    mylistvariablesy = ['d_len_xy_ML','sig_vert_ML','delta_mass_KK_ML',"delta_mass_KK_ML","sig_vert_ML","d_len_xy_ML","cos_PiKPhi_3_ML","sig_vert_ML","pt_cand_ML"]
  if (case=="Lc"):
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML']
    mylistvariablesy = ['d_len_xy_ML','sig_vert_ML']

  return mylistvariablesx,mylistvariablesy

def getgridsearchparameters(case):
  if (case=="Ds" or case=="Lc"):
    namesCV=["Random_Forest","GradientBoostingClassifier"]
    classifiersCV=[RandomForestClassifier(),GradientBoostingClassifier()]
    param_gridCV = [[{'n_estimators': [3, 10, 50, 100], 'max_features': [2,4,6,8],'max_depth': [1,4]}],[{'learning_rate': [0.01,0.05, 0.1], 'n_estimators': [1000, 2000, 5000],'max_depth' : [1, 2, 4]}]]
    changeparameter=["n_estimators","n_estimators"]
  return namesCV,classifiersCV,param_gridCV,changeparameter
  

def getDataMCfiles(case):
  if (case=="Ds"):
    fileData="/Users/gianmicheleinnocenti/MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased_skimmed.root"
    fileMC="/Users/gianmicheleinnocenti/MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased_skimmed.root"
  if (case=="Lc"):
    fileData="/Users/gianmicheleinnocenti/MLproductions/AnalysisResults_Lambdac_Data_CandBased_skimmed.root"
    fileMC="/Users/gianmicheleinnocenti/MLproductions/AnalysisResults_Lambdac_MC_CandBased_skimmed.root"
  return fileData,fileMC

def getTreeName(case):
  if (case=="Ds"):
    treename="fTreeDsFlagged"
  if (case=="Lc"):
    treename="fTreeLcFlagged"
  return treename

def getdataframe(filename,treename,variables):
  file = uproot.open(filename)
  tree = file[treename]
  dataframe=tree.pandas.df(preparestringforuproot(variables))
  return dataframe

def prepareMLsample(case,dataframe_data,dataframe_MC,nevents,option="old"):
  if (case=="Ds"):
    fmassmin=1.80
    fmassmax=2.04
    
  if (case=="Lc"):
    fmassmin=1.80
    fmassmax=2.04
  
  print  (list(dataframe_MC))
  signal_var=dataframe_MC["cand_type_ML"]
  print (("Initial n. events MC before cuts on signal: %d" % (len(dataframe_MC))))
  cand_type_ML_int=signal_var.astype(int).values
  signal_ML_array=[]
  if (option=="new"):
    signal_ML_array=((cand_type_ML_int>>3)&0b1) & ((cand_type_ML_int>>1)&0b1) | ((cand_type_ML_int>>4)&0b1) & ((cand_type_ML_int>>1)&0b1)
  if (option=="old"):
    signal_ML_array=((cand_type_ML_int==3) | (cand_type_ML_int==3))
    
  signal_ML = pd.Series(signal_ML_array)
  dataframe_MC["signal_ML"]=signal_ML
  dataframe_MC=dataframe_MC.loc[dataframe_MC["signal_ML"] == 1]
  print (("Initial n. events after the cuts on signal: %d" % (len(dataframe_MC))))
  dataframe_data["signal_ML"]=0
  
  dataframe_MC=dataframe_MC[:nevents]
  dataframe_data=dataframe_data[:nevents]
  dataframe_ML_joined = pd.concat([dataframe_MC, dataframe_data])

  print (("Events MC selected: %d" % (len(dataframe_MC))))
  print (("Events data selected: %d" % (len(dataframe_data))))
  
  if ((nevents>len(dataframe_MC)) or (nevents>len(dataframe_data))):
    print ("------------------------- ERROR: there are not so many events!!!!!! ------------------------- ")
  return dataframe_ML_joined



  
