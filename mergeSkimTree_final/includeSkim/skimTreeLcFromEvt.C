#include <iostream>
#include <vector>
#include <algorithm>
#include "tree_Dplus.C"

using namespace std;

void skimTreeLcFromEvt(TString input="AnalysisResults.root",TString output="test.root",TString ttreeout="tree_Dplus"){

cout << "NOT YET READY. NEED TO ADD ALL VARIABLES FOR Lc!!" << endl;
return;
    
  TFile *f = TFile::Open(input.Data());
  TDirectory * dir = (TDirectory*)f->Get("PWGHF_TreeCreator");
//  TList *list= (TList*)dir->Get("coutputTreeHFTreeCreator");
//  TTree*tree = (TTree*)list->FindObject(ttreeout.Data());
  TTree*tree = (TTree*)dir->Get(ttreeout.Data());
    
  tree_Dplus t(tree);
  int nevt = t.GetEntriesFast();
  cout << "RUNNING " << endl;
  TFile *fout = new TFile(output.Data(),"recreate"); 
  TTree* fTreeDplusML = new TTree("fTreeDplusFlagged","fTreeDplusFlagged");
  
  float inv_mass_ML, pt_cand_ML, d_len_ML, d_len_xy_ML, norm_dl_xy_ML, cos_p_ML, cos_p_xy_ML, imp_par_xy_ML, sig_vert_ML, max_norm_d0d0exp_ML;
  float cand_type_ML, y_cand_ML, eta_cand_ML, phi_cand_ML;
  float imp_par_prong0_ML, imp_par_prong1_ML, imp_par_prong2_ML, p_prong0_ML, p_prong1_ML, p_prong2_ML, pt_prong0_ML, pt_prong1_ML, pt_prong2_ML, eta_prong0_ML, eta_prong1_ML, eta_prong2_ML, phi_prong0_ML, phi_prong1_ML, phi_prong2_ML;
  float nTPCcls_prong0_ML, nTPCclspid_prong0_ML, nTPCcrossrow_prong0_ML, chi2perndf_prong0_ML, nITScls_prong0_ML, ITSclsmap_prong0_ML, nTPCcls_prong1_ML, nTPCclspid_prong1_ML, nTPCcrossrow_prong1_ML, chi2perndf_prong1_ML, nITScls_prong1_ML, ITSclsmap_prong1_ML, nTPCcls_prong2_ML, nTPCclspid_prong2_ML, nTPCcrossrow_prong2_ML, chi2perndf_prong2_ML, nITScls_prong2_ML, ITSclsmap_prong2_ML;
  float nsigTPC_Pi_0_ML, nsigTPC_K_0_ML, nsigTOF_Pi_0_ML, nsigTOF_K_0_ML, dEdxTPC_0_ML, ToF_0_ML, pTPC_prong0_ML, pTOF_prong0_ML, trlen_prong0_ML, start_time_res_prong0_ML, nsigTPC_Pi_1_ML, nsigTPC_K_1_ML, nsigTOF_Pi_1_ML, nsigTOF_K_1_ML, dEdxTPC_1_ML, ToF_1_ML, pTPC_prong1_ML, pTOF_prong1_ML, trlen_prong1_ML, start_time_res_prong1_ML, nsigTPC_Pi_2_ML, nsigTPC_K_2_ML, nsigTOF_Pi_2_ML, nsigTOF_K_2_ML, dEdxTPC_2_ML, ToF_2_ML, pTPC_prong2_ML, pTOF_prong2_ML, trlen_prong2_ML, start_time_res_prong2_ML;
  
  fTreeDplusML->Branch("inv_mass_ML",&inv_mass_ML,"inv_mass_ML/F");
  fTreeDplusML->Branch("pt_cand_ML",&pt_cand_ML,"pt_cand_ML/F");
  fTreeDplusML->Branch("d_len_ML",&d_len_ML,"d_len_ML/F");
  fTreeDplusML->Branch("d_len_xy_ML",&d_len_xy_ML,"d_len_xy_ML/F");
  fTreeDplusML->Branch("norm_dl_xy_ML",&norm_dl_xy_ML,"norm_dl_xy_ML/F");
  fTreeDplusML->Branch("cos_p_ML",&cos_p_ML,"cos_p_ML/F");
  fTreeDplusML->Branch("cos_p_xy_ML",&cos_p_xy_ML,"cos_p_xy_ML/F");
  fTreeDplusML->Branch("imp_par_xy_ML",&imp_par_xy_ML,"imp_par_xy_ML/F");
  fTreeDplusML->Branch("sig_vert_ML",&sig_vert_ML,"sig_vert_ML/F");
  fTreeDplusML->Branch("max_norm_d0d0exp_ML",&max_norm_d0d0exp_ML,"max_norm_d0d0exp_ML/F");

  fTreeDplusML->Branch("cand_type_ML",&cand_type_ML,"cand_type_ML/F");
  fTreeDplusML->Branch("y_cand_ML",&y_cand_ML,"y_cand_ML/F");
  fTreeDplusML->Branch("eta_cand_ML",&eta_cand_ML,"eta_cand_ML/F");
  fTreeDplusML->Branch("phi_cand_ML",&phi_cand_ML,"phi_cand_ML/F");

  fTreeDplusML->Branch("imp_par_prong0_ML",&imp_par_prong0_ML,"imp_par_prong0_ML/F");
  fTreeDplusML->Branch("imp_par_prong1_ML",&imp_par_prong1_ML,"imp_par_prong1_ML/F");
  fTreeDplusML->Branch("imp_par_prong2_ML",&imp_par_prong2_ML,"imp_par_prong2_ML/F");
  fTreeDplusML->Branch("pt_prong0_ML",&pt_prong0_ML,"pt_prong0_ML/F");
  fTreeDplusML->Branch("pt_prong1_ML",&pt_prong1_ML,"pt_prong1_ML/F");
  fTreeDplusML->Branch("pt_prong2_ML",&pt_prong2_ML,"pt_prong2_ML/F");
  fTreeDplusML->Branch("p_prong0_ML",&p_prong0_ML,"p_prong0_ML/F");
  fTreeDplusML->Branch("p_prong1_ML",&p_prong1_ML,"p_prong1_ML/F");
  fTreeDplusML->Branch("p_prong2_ML",&p_prong2_ML,"p_prong2_ML/F");
  fTreeDplusML->Branch("eta_prong0_ML",&eta_prong0_ML,"eta_prong0_ML/F");
  fTreeDplusML->Branch("eta_prong1_ML",&eta_prong1_ML,"eta_prong1_ML/F");
  fTreeDplusML->Branch("eta_prong2_ML",&eta_prong2_ML,"eta_prong2_ML/F");
  fTreeDplusML->Branch("phi_prong0_ML",&phi_prong0_ML,"phi_prong0_ML/F");
  fTreeDplusML->Branch("phi_prong1_ML",&phi_prong1_ML,"phi_prong1_ML/F");
  fTreeDplusML->Branch("phi_prong2_ML",&phi_prong2_ML,"phi_prong2_ML/F");
/*TODO, add singletrack and PID variables. Below some remaining from a previous version of Ds
  fTreeDplusML->Branch("pTPC_prong0_ML",&pTPC_prong0_ML,"pTPC_prong0_ML/F");
  fTreeDplusML->Branch("pTPC_prong1_ML",&pTPC_prong1_ML,"pTPC_prong1_ML/F");
  fTreeDplusML->Branch("pTPC_prong2_ML",&pTPC_prong2_ML,"pTPC_prong2_ML/F");
  fTreeDplusML->Branch("nTPCclspid_prong0_ML",&nTPCclspid_prong0_ML,"nTPCclspid_prong0_ML/F");
  fTreeDplusML->Branch("nTPCclspid_prong1_ML",&nTPCclspid_prong1_ML,"nTPCclspid_prong1_ML/F");
  fTreeDplusML->Branch("nTPCclspid_prong2_ML",&nTPCclspid_prong2_ML,"nTPCclspid_prong2_ML/F");
  fTreeDplusML->Branch("dEdxTPC_0_ML",&dEdxTPC_0_ML,"dEdxTPC_0_ML/F");
  fTreeDplusML->Branch("dEdxTPC_1_ML",&dEdxTPC_1_ML,"dEdxTPC_1_ML/F");
  fTreeDplusML->Branch("dEdxTPC_2_ML",&dEdxTPC_2_ML,"dEdxTPC_2_ML/F");
*/
    
  std::cout<<"nevents"<<nevt<<std::endl;
  for(Long64_t jentry=0; jentry<nevt;jentry++){
    t.GetEntry(jentry);   
    if(jentry%1000==0) cout<<jentry<<endl;
    for(int icand = 0; icand < t.n_cand; icand++){ 
      inv_mass_ML=t.inv_mass -> at(icand);
      pt_cand_ML=t.pt_cand -> at(icand);
      d_len_ML=t.d_len -> at(icand);
      d_len_xy_ML=t.d_len_xy -> at(icand);
      norm_dl_xy_ML=t.norm_dl_xy -> at(icand);
      cos_p_ML=t.cos_p -> at(icand);
      cos_p_xy_ML=t.cos_p_xy -> at(icand);
      imp_par_xy_ML=t.imp_par_xy -> at(icand);
      sig_vert_ML=t.sig_vert -> at(icand);
      max_norm_d0d0exp_ML=t.max_norm_d0d0exp -> at(icand);
        
      cand_type_ML=t.cand_type -> at(icand);
      y_cand_ML=t.y_cand -> at(icand);
      eta_cand_ML=t.eta_cand -> at(icand);
      phi_cand_ML=t.phi_cand -> at(icand);

      pt_prong0_ML=t.pt_prong0 -> at(icand);
      pt_prong1_ML=t.pt_prong1 -> at(icand);
      pt_prong2_ML=t.pt_prong2 -> at(icand);
      imp_par_prong0_ML=t.imp_par_prong0 -> at(icand);
      imp_par_prong1_ML=t.imp_par_prong1 -> at(icand);
      imp_par_prong2_ML=t.imp_par_prong2 -> at(icand);
      p_prong0_ML=t.p_prong0 -> at(icand);
      p_prong1_ML=t.p_prong1 -> at(icand);
      p_prong2_ML=t.p_prong2 -> at(icand);
      eta_prong0_ML=t.eta_prong0 -> at(icand);
      eta_prong1_ML=t.eta_prong1 -> at(icand);
      eta_prong2_ML=t.eta_prong2 -> at(icand);
      phi_prong0_ML=t.phi_prong0 -> at(icand);
      phi_prong1_ML=t.phi_prong1 -> at(icand);
      phi_prong2_ML=t.phi_prong2 -> at(icand);

      /*TODO, add singletrack and PID variables. Below some remaining from a previous version of Ds
      pTPC_prong0_ML=t.pTPC_prong0 -> at(icand);
      pTPC_prong1_ML=t.pTPC_prong1 -> at(icand);
      pTPC_prong2_ML=t.pTPC_prong2 -> at(icand);
      nTPCclspid_prong0_ML=t.nTPCclspid_prong0 -> at(icand);
      nTPCclspid_prong1_ML=t.nTPCclspid_prong1 -> at(icand);
      nTPCclspid_prong2_ML=t.nTPCclspid_prong2 -> at(icand);
      dEdxTPC_0_ML=t.dEdxTPC_0 -> at(icand);
      dEdxTPC_1_ML=t.dEdxTPC_1 -> at(icand);
      dEdxTPC_2_ML=t.dEdxTPC_2 -> at(icand);
      */
      fTreeDplusML->Fill();
    }
  }
  fout->Write();
  fout->Close();
}


int main(int argc, char *argv[])
{
  if((argc != 4))
  {
    std::cout << "Wrong number of inputs" << std::endl;
    return 1;
  }
  
  if(argc == 4)
    skimTreeDplusFromEvt(argv[1],argv[2],argv[3]);
  return 0;
}
