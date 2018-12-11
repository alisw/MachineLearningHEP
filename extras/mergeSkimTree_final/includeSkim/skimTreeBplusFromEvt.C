#include <iostream>
#include <vector>
#include <algorithm>
#include "tree_Bplus.C"
using namespace std;

void skimTreeBplusFromEvt(TString input="AnalysisResults-8.root",TString output="test.root",TString ttreeout="tree_Bplus"){

  cout << "NOT YET READY. NEED TO ADD ALL VARIABLES FOR Bplus!!" << endl;
  return;

  TFile *f = TFile::Open(input.Data());
  TDirectory * dir = (TDirectory*)f->Get("PWGHF_TreeCreator");
  TList *list= (TList*)dir->Get("coutputTreeHFTreeCreator");
  TTree*tree = (TTree*)list->FindObject(ttreeout.Data());

  tree_Bplus t(tree);
  int nevt = t.GetEntriesFast();
  cout << "RUNNING " << endl;
  TFile *fout = new TFile(output.Data(),"recreate"); 
  TTree* fTreeBplus = new TTree("fTreeBplusFlagged","fTreeBplusFlagged");
  
  float inv_mass_ML,pt_cand_ML,d_len_ML,d_len_xy_ML,norm_dl_xy_ML,cos_p_ML,cos_p_xy_ML,imp_par_ML,imp_par_xy_ML,pt_prong0_ML,pt_prong1_ML,pt_prong2_ML;
  float cand_type_ML;
  float pTPC_prong0_ML,pTPC_prong1_ML,pTPC_prong2_ML,nTPCclspid_prong0_ML,nTPCclspid_prong1_ML,nTPCclspid_prong2_ML,dEdxTPC_0_ML,dEdxTPC_1_ML,dEdxTPC_2_ML;
  int event_ID_ML;

  fTreeBplus->Branch("inv_mass_ML",&inv_mass_ML,"inv_mass_ML/F");
  fTreeBplus->Branch("pt_cand_ML",&pt_cand_ML,"pt_cand_ML/F");
  fTreeBplus->Branch("d_len_ML",&d_len_ML,"d_len_ML/F");
  fTreeBplus->Branch("d_len_xy_ML",&d_len_xy_ML,"d_len_xy_ML/F");
  fTreeBplus->Branch("norm_dl_xy_ML",&norm_dl_xy_ML,"norm_dl_xy_ML/F");
  fTreeBplus->Branch("cos_p_ML",&cos_p_ML,"cos_p_ML/F");
  fTreeBplus->Branch("cos_p_xy_ML",&cos_p_xy_ML,"cos_p_xy_ML/F");
  fTreeBplus->Branch("imp_par_xy_ML",&imp_par_xy_ML,"imp_par_xy_ML/F");
  fTreeBplus->Branch("pt_prong0_ML",&pt_prong0_ML,"pt_prong0_ML/F");
  fTreeBplus->Branch("pt_prong1_ML",&pt_prong1_ML,"pt_prong1_ML/F");
  fTreeBplus->Branch("pt_prong2_ML",&pt_prong2_ML,"pt_prong2_ML/F");
  fTreeBplus->Branch("cand_type_ML",&cand_type_ML,"cand_type_ML/F");
  fTreeBplus->Branch("pTPC_prong0_ML",&pTPC_prong0_ML,"pTPC_prong0_ML/F");
  fTreeBplus->Branch("pTPC_prong1_ML",&pTPC_prong1_ML,"pTPC_prong1_ML/F");
  fTreeBplus->Branch("pTPC_prong2_ML",&pTPC_prong2_ML,"pTPC_prong2_ML/F");
  fTreeBplus->Branch("nTPCclspid_prong0_ML",&nTPCclspid_prong0_ML,"nTPCclspid_prong0_ML/F");
  fTreeBplus->Branch("nTPCclspid_prong1_ML",&nTPCclspid_prong1_ML,"nTPCclspid_prong1_ML/F");
  fTreeBplus->Branch("nTPCclspid_prong2_ML",&nTPCclspid_prong2_ML,"nTPCclspid_prong2_ML/F");
  fTreeBplus->Branch("dEdxTPC_0_ML",&dEdxTPC_0_ML,"dEdxTPC_0_ML/F");
  fTreeBplus->Branch("dEdxTPC_1_ML",&dEdxTPC_1_ML,"dEdxTPC_1_ML/F");
  fTreeBplus->Branch("dEdxTPC_2_ML",&dEdxTPC_2_ML,"dEdxTPC_2_ML/F");
    
  fTreeBplus->Branch("event_ID_ML",&event_ID_ML,"event_ID_ML/I");

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
      pt_prong0_ML=t.pt_prong0 -> at(icand);
      pt_prong1_ML=t.pt_prong1 -> at(icand);
      pt_prong2_ML=t.pt_prong2 -> at(icand);
      cand_type_ML=0;
      std::cout<<t.cand_type -> at(icand)<<std::endl;
      if ((t.cand_type -> at(icand))==9) cand_type_ML=2;
      pTPC_prong0_ML=t.pTPC_prong0 -> at(icand);
      pTPC_prong1_ML=t.pTPC_prong0 -> at(icand);
      pTPC_prong2_ML=t.pTPC_prong0 -> at(icand);
      nTPCclspid_prong0_ML=t.nTPCclspid_prong0 -> at(icand);
      nTPCclspid_prong1_ML=t.nTPCclspid_prong1 -> at(icand);
      nTPCclspid_prong2_ML=t.nTPCclspid_prong2 -> at(icand);
      dEdxTPC_0_ML=t.dEdxTPC_0 -> at(icand);
      dEdxTPC_1_ML=t.dEdxTPC_1 -> at(icand);
      dEdxTPC_2_ML=t.dEdxTPC_2 -> at(icand);

      event_ID_ML=jentry;

      fTreeBplus->Fill();
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
    skimTreeBplusFromEvt(argv[1],argv[2],argv[3]);
  return 0;
}
