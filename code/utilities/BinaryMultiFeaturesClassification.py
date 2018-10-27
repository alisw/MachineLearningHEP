from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from utilitiesGeneral import filterdataframe_pt,splitdataframe_sigbkg,checkdir,preparestringforuproot
import pandas as pd
import uproot

def getvariablestraining(case):
  if (case=="Ds"):
    mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"]
  return mylistvariables

def getvariablesothers(case):
  if (case=="Ds"):
    mylistvariablesothers=['inv_mass_ML','pt_cand_ML']
  return mylistvariablesothers

def getvariableissignal(case):
  if (case=="Ds"):
    myvariablesy='signal_ML'
  return myvariablesy

def getvariablesall(case):
  if (case=="Ds"):
    mylistvariablesall=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML",'inv_mass_ML','pt_cand_ML','signal_ML',"cand_type_ML"]
  return mylistvariablesall

def getvariablecorrelation(case):
  if (case=="Ds"):
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML','sig_vert_ML',"pt_cand_ML","pt_cand_ML","norm_dl_xy_ML","cos_PiDs_ML","cos_p_xy_ML","cos_p_xy_ML"]
    mylistvariablesy = ['d_len_xy_ML','sig_vert_ML','delta_mass_KK_ML',"delta_mass_KK_ML","sig_vert_ML","d_len_xy_ML","cos_PiKPhi_3_ML","sig_vert_ML","pt_cand_ML"]
  return mylistvariablesx,mylistvariablesy

def getgridsearchparameters(case):
  if (case=="Ds"):
    namesCV=["Random_Forest","GradientBoostingClassifier"]
    classifiersCV=[RandomForestClassifier(),GradientBoostingClassifier()]
    param_gridCV = [[{'n_estimators': [3, 10, 50, 100], 'max_features': [2,4,6,8],'max_depth': [1,4]}],[{'learning_rate': [0.01,0.05, 0.1], 'n_estimators': [1000, 2000, 5000],'max_depth' : [1, 2, 4]}]]
    changeparameter=["n_estimators","n_estimators"]
  return namesCV,classifiersCV,param_gridCV,changeparameter
  

def getDataMCfiles(case):
  if (case=="Ds"):
    fileData="../buildsampleEventBased/rootfiles/LHC17p_FAST_run282343_AnalysisResultsData_CandBased.root"
    fileMC="../buildsampleEventBased/rootfiles/LHC18a4a2_fast_run282343_AnalysisResultsDmesonsMC_CandBased.root"
  return fileData,fileMC

def getTreeName(case):
  if (case=="Ds"):
    treename="fTreeDsFlagged"
  return treename

def getdataframe(filename,treename,variables):
  file = uproot.open(filename)
  tree = file[treename]
  dataframe=tree.pandas.df(preparestringforuproot(variables))
  return dataframe

def prepareMLsample(case,dataframe_data,dataframe_MC,nevents):
  if (case=="Ds"):
    fmassmin=1.80
    fmassmax=2.04
  
    signal_var=dataframe_MC["cand_type_ML"]
    cand_type_ML_int=signal_var.astype(int).values
    signal_ML_array=((cand_type_ML_int>>3)&0b1) & ((cand_type_ML_int>>1)&0b1)
    signal_ML = pd.Series(signal_ML_array)
    dataframe_MC["signal_ML"]=signal_ML
    dataframe_MC=dataframe_MC.loc[dataframe_MC["signal_ML"] == 1]
    dataframe_data=dataframe_data.loc[(dataframe_data["inv_mass_ML"]<fmassmin) | (dataframe_data["inv_mass_ML"]>fmassmax)]
    dataframe_data["signal_ML"]=0
    
  dataframe_MC=dataframe_MC[:nevents]
  dataframe_data=dataframe_data[:nevents]
  dataframe_ML_joined = pd.concat([dataframe_MC, dataframe_data])
    
  return dataframe_ML_joined








  
