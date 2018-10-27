#include <iostream>
#include <vector>
#include <algorithm>
#include "tree_Ds.C"

using namespace std;

void makeNtupleCandBased(TString input="AnalysisResults-8.root",TString output="test.root",TString ttreeout="tree_Ds"){

  TFile *f = TFile::Open(input.Data());
  TDirectory * dir = (TDirectory*)f->Get("PWGHF_TreeCreator");
  TList *list= (TList*)dir->Get("coutputTreeHFTreeCreator");
  TTree*tree = (TTree*)list->FindObject(ttreeout.Data());

  tree_Ds t(tree);
  int nevt = t.GetEntriesFast();
  cout << nevt << endl;
  TFile *fout = new TFile(output.Data(),"recreate"); 
  TTree* fTreeDsML = new TTree("fTreeDsFlagged","fTreeDsFlagged");
  
  float inv_mass_ML,pt_cand_ML,d_len_ML,d_len_xy_ML,norm_dl_xy_ML,cos_p_ML,cos_p_xy_ML,imp_par_ML,imp_par_xy_ML,pt_prong0_ML,pt_prong1_ML,pt_prong2_ML,sig_vert_ML,delta_mass_KK_ML,cos_PiDs_ML,cos_PiKPhi_3_ML;
  float cand_type_ML;
  float pTPC_prong0_ML,pTPC_prong1_ML,pTPC_prong2_ML,nTPCclspid_prong0_ML,nTPCclspid_prong1_ML,nTPCclspid_prong2_ML,dEdxTPC_0_ML,dEdxTPC_1_ML,dEdxTPC_2_ML;
  
  fTreeDsML->Branch("inv_mass_ML",&inv_mass_ML,"inv_mass_ML/F");
  fTreeDsML->Branch("pt_cand_ML",&pt_cand_ML,"pt_cand_ML/F");
  fTreeDsML->Branch("d_len_ML",&d_len_ML,"d_len_ML/F");
  fTreeDsML->Branch("d_len_xy_ML",&d_len_xy_ML,"d_len_xy_ML/F");
  fTreeDsML->Branch("norm_dl_xy_ML",&norm_dl_xy_ML,"norm_dl_xy_ML/F");
  fTreeDsML->Branch("cos_p_ML",&cos_p_ML,"cos_p_ML/F");
  fTreeDsML->Branch("cos_p_xy_ML",&cos_p_xy_ML,"cos_p_xy_ML/F");
  fTreeDsML->Branch("imp_par_xy_ML",&imp_par_xy_ML,"imp_par_xy_ML/F");
  fTreeDsML->Branch("pt_prong0_ML",&pt_prong0_ML,"pt_prong0_ML/F");
  fTreeDsML->Branch("pt_prong1_ML",&pt_prong1_ML,"pt_prong1_ML/F");
  fTreeDsML->Branch("pt_prong2_ML",&pt_prong2_ML,"pt_prong2_ML/F");
  fTreeDsML->Branch("sig_vert_ML",&sig_vert_ML,"sig_vert_ML/F");
  fTreeDsML->Branch("delta_mass_KK_ML",&delta_mass_KK_ML,"delta_mass_KK_ML/F");
  fTreeDsML->Branch("cos_PiDs_ML",&cos_PiDs_ML,"cos_PiDs_ML/F");
  fTreeDsML->Branch("cos_PiKPhi_3_ML",&cos_PiKPhi_3_ML,"cos_PiKPhi_3_ML/F");
  fTreeDsML->Branch("cand_type_ML",&cand_type_ML,"cand_type_ML/F");
  fTreeDsML->Branch("pTPC_prong0_ML",&pTPC_prong0_ML,"pTPC_prong0_ML/F");
  fTreeDsML->Branch("pTPC_prong1_ML",&pTPC_prong1_ML,"pTPC_prong1_ML/F");
  fTreeDsML->Branch("pTPC_prong2_ML",&pTPC_prong2_ML,"pTPC_prong2_ML/F");
  fTreeDsML->Branch("nTPCclspid_prong0_ML",&nTPCclspid_prong0_ML,"nTPCclspid_prong0_ML/F");
  fTreeDsML->Branch("nTPCclspid_prong1_ML",&nTPCclspid_prong1_ML,"nTPCclspid_prong1_ML/F");
  fTreeDsML->Branch("nTPCclspid_prong2_ML",&nTPCclspid_prong2_ML,"nTPCclspid_prong2_ML/F");
  fTreeDsML->Branch("dEdxTPC_0_ML",&dEdxTPC_0_ML,"dEdxTPC_0_ML/F");
  fTreeDsML->Branch("dEdxTPC_1_ML",&dEdxTPC_0_ML,"dEdxTPC_1_ML/F");
  fTreeDsML->Branch("dEdxTPC_2_ML",&dEdxTPC_0_ML,"dEdxTPC_2_ML/F");


  for(Long64_t jentry=0; jentry<nevt;jentry++){
    t.GetEntry(jentry);   
    for(int icand = 0; icand < t.n_cand; icand++){ 
      inv_mass_ML=t.inv_mass -> at(icand);
      pt_cand_ML=t.pt_cand -> at(icand);
      d_len_ML=t.d_len -> at(icand);
      d_len_xy_ML=t.d_len_xy -> at(icand);
      norm_dl_xy_ML=t.norm_dl_xy -> at(icand);
      cos_p_ML=t.cos_p -> at(icand);
      cos_p_xy_ML=t.cos_p_xy -> at(icand);
      imp_par_xy_ML=t.imp_par_xy -> at(icand);
      pt_prong0_ML=t.pt_prong0 -> at(icand);
      pt_prong1_ML=t.pt_prong1 -> at(icand);
      pt_prong2_ML=t.pt_prong2 -> at(icand);
      sig_vert_ML=t.sig_vert -> at(icand);
      delta_mass_KK_ML=t.delta_mass_KK -> at(icand);
      cos_PiDs_ML=t.cos_PiDs -> at(icand);
      cos_PiKPhi_3_ML=t.cos_PiKPhi_3 -> at(icand);
      cand_type_ML=t.cand_type -> at(icand);
      
      pTPC_prong0_ML=t.pTPC_prong0 -> at(icand);
      pTPC_prong1_ML=t.pTPC_prong0 -> at(icand);
      pTPC_prong2_ML=t.pTPC_prong0 -> at(icand);
      nTPCclspid_prong0_ML=t.nTPCclspid_prong0 -> at(icand);
      nTPCclspid_prong1_ML=t.nTPCclspid_prong1 -> at(icand);
      nTPCclspid_prong2_ML=t.nTPCclspid_prong2 -> at(icand);
      dEdxTPC_0_ML=t.dEdxTPC_0 -> at(icand);
      dEdxTPC_1_ML=t.dEdxTPC_1 -> at(icand);
      dEdxTPC_2_ML=t.dEdxTPC_2 -> at(icand);

      fTreeDsML->Fill();
    }
  }
  fout->Write();
  fout->Close();
  return true; 
}
