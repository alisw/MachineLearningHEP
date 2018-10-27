#include <iostream>
#include <vector>
#include <algorithm>
#include "tree_Ds.C"

using namespace std;

void test(TString foutput="test.root",int maxevents=1000){
  tree_Ds * t = new tree_Ds();

  TFile *fout = new TFile(foutput.Data(),"recreate"); 
  TTree* fTreeDsML = new TTree("fTreeDsFlagged","fTreeDsFlagged");
  
  float inv_mass_ML,pt_cand_ML,d_len_ML,d_len_xy_ML,norm_dl_xy_ML,cos_p_ML,cos_p_xy_ML,imp_par_ML,imp_par_xy_ML,pt_prong0_ML,pt_prong1_ML,pt_prong2_ML,sig_vert_ML,delta_mass_KK_ML,cos_PiDs_ML,cos_PiKPhi_3_ML;
  float signal_ML,selected_std_ML;
  
  fTreeDsML->Branch("inv_mass_ML",&inv_mass_ML,"inv_mass_ML/F");
  fTreeDsML->Branch("pt_cand_ML",&pt_cand_ML,"pt_cand_ML/F");
  fTreeDsML->Branch("d_len_ML",&d_len_ML,"d_len_ML/F");
  fTreeDsML->Branch("d_len_xy_ML",&d_len_xy_ML,"d_len_xy_ML/F");
  fTreeDsML->Branch("norm_dl_xy_ML",&norm_dl_xy_ML,"norm_dl_xy_ML/F");
  fTreeDsML->Branch("cos_p_ML",&cos_p_ML,"cos_p_ML/F");
  fTreeDsML->Branch("cos_p_xy_ML",&cos_p_xy_ML,"cos_p_xy_ML/F");
  fTreeDsML->Branch("imp_par_ML",&imp_par_ML,"imp_par_ML/F");
  fTreeDsML->Branch("imp_par_xy_ML",&imp_par_xy_ML,"imp_par_xy_ML/F");
  fTreeDsML->Branch("pt_prong0_ML",&pt_prong0_ML,"pt_prong0_ML/F");
  fTreeDsML->Branch("pt_prong1_ML",&pt_prong1_ML,"pt_prong1_ML/F");
  fTreeDsML->Branch("pt_prong2_ML",&pt_prong2_ML,"pt_prong2_ML/F");
  fTreeDsML->Branch("sig_vert_ML",&sig_vert_ML,"sig_vert_ML/F");
  fTreeDsML->Branch("delta_mass_KK_ML",&delta_mass_KK_ML,"delta_mass_KK_ML/F");
  fTreeDsML->Branch("cos_PiDs_ML",&cos_PiDs_ML,"cos_PiDs_ML/F");
  fTreeDsML->Branch("cos_PiKPhi_3_ML",&cos_PiKPhi_3_ML,"cos_PiKPhi_3_ML/F");
  fTreeDsML->Branch("signal_ML",&signal_ML,"signal_ML/F");
  fTreeDsML->Branch("selected_std_ML",&selected_std_ML,"selected_std_ML/F");

  Long64_t nevt = t->GetEntriesFast();
  for (Long64_t jentry=0; jentry<nevt;jentry++) {
    t -> GetEntry(jentry);
    std::cout<<t->n_cand<<std::endl;
    }
  }
