#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
using namespace std;
#include <iostream>

bool skimTreeLambdac(TString finput,TString foutput,TString treename, int maxevents=-1){

  // build the signalsample ntuple
  TFile *fin = new TFile(finput.Data()); 
  TTree *fTreeLc = (TTree*)fin->Get(treename.Data()); 
  float inv_mass,pt_cand,d_len,d_len_xy,norm_dl_xy,cos_p,imp_par,imp_par_xy,sig_vert,dca,dist_12,pt_p,pt_K,pt_pi;
  float cand_type;

  fTreeLc->SetBranchAddress("InvMass",&inv_mass);
  fTreeLc->SetBranchAddress("PtLc",&pt_cand);
  fTreeLc->SetBranchAddress("DecayL",&d_len);
  fTreeLc->SetBranchAddress("DecayLXY",&d_len_xy);
  fTreeLc->SetBranchAddress("DecayLXYSig",&norm_dl_xy);
  fTreeLc->SetBranchAddress("Dist12",&dist_12);
  fTreeLc->SetBranchAddress("CosP",&cos_p);
  fTreeLc->SetBranchAddress("Ptp",&pt_p);
  fTreeLc->SetBranchAddress("PtK",&pt_K);
  fTreeLc->SetBranchAddress("Ptpi",&pt_pi);
  fTreeLc->SetBranchAddress("SigVert",&sig_vert);
  fTreeLc->SetBranchAddress("DCA",&dca);
  fTreeLc->SetBranchAddress("isLcBkg",&cand_type);

  TFile *fout = new TFile(foutput.Data(),"recreate"); 
  TTree* fTreeLcML = new TTree("fTreeLcFlagged","fTreeLcFlagged");
  
  float inv_mass_ML,pt_cand_ML,d_len_ML,d_len_xy_ML,norm_dl_xy_ML,cos_p_ML,pt_p_ML,pt_K_ML,pt_pi_ML,sig_vert_ML,dca_ML,dist_12_ML;
  float signal_ML;
  
  fTreeLcML->Branch("inv_mass_ML",&inv_mass_ML,"inv_mass_ML/F");
  fTreeLcML->Branch("pt_cand_ML",&pt_cand_ML,"pt_cand_ML/F");
  fTreeLcML->Branch("d_len_ML",&d_len_ML,"d_len_ML/F");
  fTreeLcML->Branch("d_len_xy_ML",&d_len_xy_ML,"d_len_xy_ML/F");
  fTreeLcML->Branch("norm_dl_xy_ML",&norm_dl_xy_ML,"norm_dl_xy_ML/F");
  fTreeLcML->Branch("dist_12_ML",&dist_12_ML,"dist_12_ML/F");
  fTreeLcML->Branch("cos_p_ML",&cos_p_ML,"cos_p_ML/F");
  fTreeLcML->Branch("pt_p_ML",&pt_p_ML,"pt_p_ML/F");
  fTreeLcML->Branch("pt_K_ML",&pt_K_ML,"pt_K_ML/F");
  fTreeLcML->Branch("pt_pi_ML",&pt_pi_ML,"pt_pi_ML/F");
  fTreeLcML->Branch("sig_vert_ML",&sig_vert_ML,"sig_vert_ML/F");
  fTreeLcML->Branch("dca_ML",&dca_ML,"dca_vert_ML/F");
  fTreeLcML->Branch("signal_ML",&signal_ML,"signal_ML/F");
   
  Long64_t nentries = fTreeLc->GetEntries();
  std::cout<<nentries<<std::endl;
  if (maxevents>nentries) {
    std::cout<<"the number of events in the ntupla is smaller than the n. events you want to run on"<<std::endl;
    std::cout<<"we will run on what you have"<<std::endl;
    maxevents=nentries;
  }
  if (maxevents==-1) maxevents=nentries;

  for(Long64_t i=0;i<maxevents;i++){ 
    fTreeLc->GetEntry(i);
    if(i%1000==0) cout<<i<<endl;
    inv_mass_ML=inv_mass;
    pt_cand_ML=pt_cand;
    d_len_ML=d_len;
    d_len_xy_ML=d_len_xy;
    norm_dl_xy_ML=norm_dl_xy;
    dist_12_ML=dist_12;
    cos_p_ML=cos_p;
    pt_p_ML=pt_p;
    pt_K_ML=pt_K;
    pt_pi_ML=pt_pi;
    sig_vert_ML=sig_vert;
    dca_ML=dca;
    fTreeLcML->Fill();
    } 
  fout->Write();
  fout->Close();
  return true;  
}

int main(int argc, char *argv[])
{
  if((argc != 5))
  {
    std::cout << "Wrong number of inputs" << std::endl;
    return 1;
  }
  
  if(argc == 5)
    skimTreeLambdac(argv[1],argv[2],argv[3],atoi(argv[4]));
  return 0;
}
