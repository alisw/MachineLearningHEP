###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to define: analysis type, data used, variables for training and other applications
Methods to load and prepare data for training
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier 
import pandas as pd
import uproot

def getvariablestraining(case):
  mylistvariables=[]
  if (case=="Ds"):
    mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"]
  if (case=="Dplus"):
    mylistvariables=['d_len_ML','d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML','max_norm_d0d0exp_ML']
  if (case=="Lc"):
    mylistvariables=['d_len_ML','d_len_xy_ML','norm_dl_xy_ML','dist_12_ML','cos_p_ML','pt_p_ML','pt_K_ML','pt_pi_ML','sig_vert_ML','dca_ML']
  if (case=="Bplus"):
    mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML']
  if ((case=="PIDPion") | (case=="PIDKaon")):
    mylistvariables=['dedx0_ML','tof0_ML','dca0_ML','sigdca0_ML','chisq0_ML','itscl0_ML','tpccl0_ML']
  if (case=="lightquarkjet"):
    mylistvariables=['Eta_Rec_ML','Phi_Rec_ML','Mass_Rec_ML','JetMultiplicity_Rec_ML','Angularity_ML','PTD_ML']
  if (case=="hypertritium"):
    mylistvariables=['dca', 'dl', 'cos_alpha']    
  if (case=="testregression"):
    mylistvariables=['d_len_xy_ML','cos_p_xy_ML']
  return mylistvariables

def getvariablesBoundaries(case):
  mylistvariablesboundaries=[]
  if (case=="Ds"):
    mylistvariablesboundaries=['d_len_xy_ML','delta_mass_KK_ML']
  if (case=="Dplus"):
      mylistvariablesboundaries=['d_len_ML','max_norm_d0d0exp_ML'] #not sure if this is correct
  if (case=="Lc"):
    mylistvariablesboundaries=['d_len_xy_ML','dca_ML']
  if (case=="Bplus"):
    mylistvariablesboundaries=['d_len_xy_ML','cos_p_ML']
  if ((case=="PIDPion") | (case=="PIDKaon")):
    mylistvariablesboundaries=['dedx0_ML','pdau0_ML']
  if ((case=="lightquarkjet")):
    mylistvariablesboundaries=['Angularity_ML','PTD_ML']
  if (case=="hypertritium"):
    mylistvariablesboundaries=['dca','dl']
  if (case=="testregression"):
    mylistvariablesboundaries=['delta_mass_KK_ML',"cos_p_xy_ML"]
  return mylistvariablesboundaries


def getvariablesothers(case):
  mylistvariablesothers=[]
  if (case=="Ds" or case=="Dplus" or case=="Lc" or case=="Bplus"):
    mylistvariablesothers=['inv_mass_ML','pt_cand_ML']
  if ((case=="PIDPion") | (case=="PIDKaon")):
    mylistvariablesothers=['pdau0_ML','pdg0_ML']
  if (case=="lightquarkjet"):
    mylistvariablesothers=['Pt_Rec_ML']
  if (case=="hypertritium"):
    mylistvariablesothers=['pt', 'invmass']
  if (case=="testregression"):
    mylistvariablesothers=['inv_mass_ML','pt_cand_ML']
  return mylistvariablesothers

def getvariableissignal(case):
  myvariablesy=0
  if (case=="Ds" or case=="Dplus" or case=="Lc" or case=="Bplus"):
    myvariablesy='signal_ML'
  if ((case=="PIDPion") | (case=="PIDKaon")):
    myvariablesy='signal_ML'
  if (case=="lightquarkjet"):
    myvariablesy='signal_ML'
  if (case=="hypertritium"):
    myvariablesy='signal_ML'
  if (case=="testregression"):
    myvariablesy='signal_ML'
  return myvariablesy

def getvariabletarget(case):
  myvariablestarget=0
  if (case=="Ds" or case=="Dplus" or case=="Lc" or case=="Bplus"):
    myvariablestarget='signal_ML'
  if ((case=="PIDPion") | (case=="PIDKaon")):
    myvariablestarget='signal_ML'
  if ((case=="lightquarkjet")):
    myvariablestarget='signal_ML'
  if ((case=="hypertritium")):
    myvariablestarget='signal_ML'
  if (case=="testregression"):
    myvariablestarget='norm_dl_xy_ML'
  return myvariablestarget

def getvariablesall(case):
  mylistvariablesall=[]
  if (case=="Ds"):
    mylistvariablesall=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML",'inv_mass_ML','pt_cand_ML','signal_ML',"cand_type_ML"]
  if (case=="Dplus"):
      mylistvariablesall=['d_len_ML','d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"max_norm_d0d0exp_ML",'inv_mass_ML','pt_cand_ML','signal_ML',"cand_type_ML"]
  if (case=="Lc"):
    mylistvariablesall=['inv_mass_ML','pt_cand_ML','d_len_ML','d_len_xy_ML','norm_dl_xy_ML','dist_12_ML','cos_p_ML','pt_p_ML','pt_K_ML','pt_pi_ML','sig_vert_ML','dca_ML','cand_type_ML']
  if (case=="Bplus"):
    mylistvariablesall=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','inv_mass_ML','pt_cand_ML','signal_ML',"cand_type_ML"]
  if ((case=="PIDPion") | (case=="PIDKaon")):
    mylistvariablesall=['dedx0_ML','tof0_ML','dca0_ML','sigdca0_ML','chisq0_ML','itscl0_ML','tpccl0_ML','pdau0_ML','pdg0_ML']
  if (case=="lightquarkjet"):
    mylistvariablesall=['Pt_Rec_ML','Eta_Rec_ML','Phi_Rec_ML','Mass_Rec_ML','JetMultiplicity_Rec_ML','Parton_Flag_ML','Parton_1_Flag_ML','Parton_1_Eta_ML','Parton_1_Phi_ML','Parton_2_Flag_ML','Parton_2_Eta_ML','Parton_2_Phi_ML','Angularity_ML','PTD_ML']
  if (case=="hypertritium"):
    mylistvariablesall=['dca', 'dl', 'cos_alpha','pt', 'invmass','signal']
  if (case=="testregression"):
    mylistvariablesall=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML",'inv_mass_ML','pt_cand_ML','signal_ML',"cand_type_ML"]
  return mylistvariablesall

def getvariablecorrelation(case):
  mylistvariablesx=[]
  mylistvariablesy=[]
  if (case=="Ds"):
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML','sig_vert_ML',"pt_cand_ML","pt_cand_ML","norm_dl_xy_ML","cos_PiDs_ML","cos_p_xy_ML","cos_p_xy_ML"]
    mylistvariablesy = ['d_len_xy_ML','sig_vert_ML','delta_mass_KK_ML',"delta_mass_KK_ML","sig_vert_ML","d_len_xy_ML","cos_PiKPhi_3_ML","sig_vert_ML","pt_cand_ML"]
  if (case=="Dplus"):
    #"random" combinations
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML','sig_vert_ML',"pt_cand_ML","pt_cand_ML","norm_dl_xy_ML","sig_vert_ML","cos_p_xy_ML","cos_p_xy_ML"]
    mylistvariablesy = ['d_len_xy_ML','sig_vert_ML','max_norm_d0d0exp_ML',"imp_par_xy_ML","sig_vert_ML","d_len_xy_ML","imp_par_xy_ML","sig_vert_ML","pt_cand_ML"]
  if (case=="Lc"):
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML']
    mylistvariablesy = ['d_len_xy_ML','sig_vert_ML']
  if (case=="Bplus"):
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML']
    mylistvariablesy = ['d_len_xy_ML','cos_p_ML']
  if ((case=="PIDPion") | (case=="PIDKaon")):
    mylistvariablesx = ['pdau0_ML','pdau0_ML','itscl0_ML']
    mylistvariablesy = ['dedx0_ML','tof0_ML','chisq0_ML']
  if ((case=="lightquarkjet")):
    mylistvariablesx = ['Pt_Rec_ML','Eta_Rec_ML','Phi_Rec_ML','Mass_Rec_ML','Parton_1_Flag_ML','Angularity_ML']
    mylistvariablesy = ['Eta_Rec_ML','Phi_Rec_ML','Mass_Rec_ML','JetMultiplicity_Rec_ML','Parton_2_Flag_ML','PTD_ML']
  if ((case=="hypertritium")):
    mylistvariablesx = ['dca','dl','cos_alpha','pt']
    mylistvariablesy = ['dl','cos_alpha','pt','invmass']
  if (case=="testregression"):
    mylistvariablesx = ['pt_cand_ML','d_len_xy_ML','sig_vert_ML',"pt_cand_ML","pt_cand_ML","norm_dl_xy_ML","cos_PiDs_ML","cos_p_xy_ML","cos_p_xy_ML"]
    mylistvariablesy = ['d_len_xy_ML','sig_vert_ML','delta_mass_KK_ML',"delta_mass_KK_ML","sig_vert_ML","d_len_xy_ML","cos_PiKPhi_3_ML","sig_vert_ML","pt_cand_ML"]
  return mylistvariablesx,mylistvariablesy

def getgridsearchparameters(case):
  if (case=="Ds" or case=="Dplus" or case=="Lc"):
    namesCV=["Random_Forest","GradientBoostingClassifier"]
    classifiersCV=[RandomForestClassifier(),GradientBoostingClassifier()]
    param_gridCV = [[{'n_estimators': [3, 10, 50, 100], 'max_features': [2,4,6,8],'max_depth': [1,4]}],[{'learning_rate': [0.01,0.05, 0.1], 'n_estimators': [1000, 2000, 5000],'max_depth' : [1, 2, 4]}]]
    changeparameter=["n_estimators","n_estimators"]
  return namesCV,classifiersCV,param_gridCV,changeparameter
  

def getDataMCfiles(case):
  fileData=""
  fileMC=""
  if (case=="Ds"):
    fileData="../MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased_skimmed.root"
    fileMC="../MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased_skimmed.root"
  if (case=="Dplus"):
    fileData="../MLproductions/AnalysisResults_Data_fullTrainOne7thStat1Run_Dplus_skimmed.root"
    fileMC="../MLproductions/AnalysisResults_MC_testTrainFullStat1Run_Dplus_skimmed.root"
  if (case=="Lc"):
    fileData="../MLproductions/AnalysisResults_Lambdac_Data_CandBased_skimmed.root"
    fileMC="../MLproductions/AnalysisResults_Lambdac_MC_CandBased_skimmed.root"
  if (case=="Bplus"):
    fileData="../MLproductions/AnalysisResults_TreeForBplus_MC_EventBased_skimmed.root"
    fileMC="../MLproductions/AnalysisResults_TreeForBplus_MC_EventBased_skimmed.root"
  if ((case=="PIDPion") | (case=="PIDKaon")):
    fileData="../MLproductions/AnalysisResults_TreeForPIDwithML_Dplus_CandBased_skimmed.root"
    fileMC="../MLproductions/AnalysisResults_TreeForPIDwithML_Dplus_CandBased_skimmed.root"
  if ((case=="lightquarkjet")):
    fileData="../MLproductions/AnalysisResults_TreeforJetsPythiaPP_MC_skimmed_GenMatched.root"
    fileMC="../MLproductions/AnalysisResults_TreeforJetsPythiaPP_MC_skimmed_GenMatched.root"
  if ((case=="hypertritium")):
    fileData="../MLproductions/AnalysisResultsHyper_MC.root"
    fileMC="../MLproductions/AnalysisResultsHyper_MC.root"
  if (case=="testregression"):
    fileData="../MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased_skimmed.root"
    fileMC="../MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased_skimmed.root"
  return fileData,fileMC

def getTreeName(case):
  treename=""
  if (case=="Ds"):
    treename="fTreeDsFlagged"
  if (case=="Dplus"):
    treename="fTreeDplusFlagged"
  if (case=="Lc"):
    treename="fTreeLcFlagged"
  if (case=="Bplus"):
    treename="fTreeBplusFlagged"
  if ((case=="PIDPion") | (case=="PIDKaon")):
    treename="fTreePIDFlagged"
  if (case=="lightquarkjet"):
    treename="fTreeDsFlagged"
  if (case=="hypertritium"):
    treename="tree_sim"
  if (case=="testregression"):
    treename="fTreeDsFlagged"

  return treename

def getmasscut(case):
  fmassmin=-1
  fmassmax=-1  
  if (case=="Ds"):
    fmassmin=1.85
    fmassmax=2.04
  
  if (case=="Dplus"):
    fmassmin=1.87-0.1
    fmassmax=1.87+0.1

  if (case=="Lc"):
    fmassmin=2.2864-0.1
    fmassmax=2.2864+0.1
    
  if (case=="Bplus"):
    fmassmin=5.279-0.100
    fmassmax=5.279+0.100
    
  return fmassmin,fmassmax

def getPDGcode(case):
  if (case=="PIDPion"):
    PDGcode=211
    
  if (case=="PIDKaon"):
    PDGcode=321
    
  return PDGcode

def prepareMLsample(MLtype,MLsubtype,case,dataframe_data,dataframe_MC,nevents):
  dataframe_ML_joined = pd.DataFrame()
  
  if(MLtype=="BinaryClassification" ):
    if(MLsubtype=="HFmeson"):
      dataframe_bkg=dataframe_data
      dataframe_sig=dataframe_MC
      fmassmin,fmassmax=getmasscut(case)
      #dataframe_sig=dataframe_sig.loc[(dataframe_sig["cand_type_ML"] == 2) | (dataframe_sig["cand_type_ML"] == 3)]
      dataframe_sig=dataframe_sig.loc[(dataframe_sig["cand_type_ML"] == 10) | (dataframe_sig["cand_type_ML"] == 11) |(dataframe_sig["cand_type_ML"] == 18) | (dataframe_sig["cand_type_ML"] == 19)]
      dataframe_sig['signal_ML'] = 1
      dataframe_bkg=dataframe_bkg.loc[(dataframe_bkg["inv_mass_ML"] < fmassmin) | (dataframe_bkg["inv_mass_ML"] > fmassmax)]
      dataframe_bkg['signal_ML'] = 0
    
    if(MLsubtype=="PID"):
      dataframe_MC["pdg0_ML"]=dataframe_MC["pdg0_ML"].abs()
      dataframe_sig=dataframe_MC.loc[(dataframe_MC["pdg0_ML"] == getPDGcode(case))]
      dataframe_sig['signal_ML'] = 1
      dataframe_bkg=dataframe_MC.loc[(dataframe_MC["pdg0_ML"] != getPDGcode(case))]
      dataframe_bkg['signal_ML'] = 0

    if(MLsubtype=="jettagging"):
      dataframe_bkg=dataframe_MC
      dataframe_sig=dataframe_MC
      if (case=="lightquarkjet"):
        dataframe_sig=dataframe_sig.loc[(dataframe_sig["Parton_Flag_ML"] == 1) | (dataframe_sig["Parton_Flag_ML"] == 2)| (dataframe_sig["Parton_Flag_ML"] == 3) | (dataframe_sig["Parton_Flag_ML"] == 4) | (dataframe_sig["Parton_Flag_ML"] == 5)]
        dataframe_bkg=dataframe_bkg.loc[(dataframe_bkg["Parton_Flag_ML"] > 5)] 
      dataframe_sig['signal_ML'] = 1
      dataframe_bkg['signal_ML'] = 0

    if(MLsubtype=="nuclei"):
      dataframe_bkg=dataframe_MC
      dataframe_sig=dataframe_MC
      if (case=="hypertritium"):
        dataframe_sig=dataframe_sig.loc[(dataframe_sig["signal"] == 1)]
        dataframe_bkg=dataframe_bkg.loc[(dataframe_bkg["signal"] == -1)]
      dataframe_sig['signal_ML'] = 1
      dataframe_bkg['signal_ML'] = 0

  if(MLtype=="Regression" ):
    if(MLsubtype=="test"):
      dataframe_bkg=dataframe_data
      dataframe_sig=dataframe_MC
      fmassmin,fmassmax=getmasscut("Ds")
      dataframe_sig=dataframe_sig.loc[(dataframe_sig["cand_type_ML"] == 2) | (dataframe_sig["cand_type_ML"] == 3)]
      dataframe_sig['signal_ML'] = 1
      dataframe_bkg=dataframe_bkg.loc[(dataframe_bkg["inv_mass_ML"] < fmassmin) | (dataframe_bkg["inv_mass_ML"] > fmassmax)]
      dataframe_bkg['signal_ML'] = 0

  dataframe_sig=dataframe_sig[:nevents]
  dataframe_bkg=dataframe_bkg[:nevents]
  dataframe_ML_joined = pd.concat([dataframe_sig, dataframe_bkg])
  if ((nevents>len(dataframe_sig)) or (nevents>len(dataframe_bkg))):
    print ("------------------------- ERROR: there are not so many events!!!!!! ------------------------- ")
      
  return dataframe_ML_joined,dataframe_sig,dataframe_bkg



  
