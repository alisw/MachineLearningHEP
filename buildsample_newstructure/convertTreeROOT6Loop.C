#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
using namespace std;
double fMassDs=1.967;
double fMassDsRange=0.05;
#include <iostream>

bool convertTreeROOT6Loop(TString finput,TString treename,TString foutput,int maxevents){

  // build the signalsample ntuple
  TFile *fin = new TFile(finput.Data()); 
  TDirectory *dir=(TDirectory*)fin->Get("PWGHF_TreeCreator");
  TList *list=(TList*)dir->Get("coutputTreeHFTreeCreator");
  TTree *fTreeDs = (TTree*)list->FindObject(treename.Data()); 
  float inv_mass,pt_cand,d_len,d_len_xy,norm_dl_xy,cos_p,cos_p_xy,imp_par,imp_par_xy,pt_prong0,pt_prong1,pt_prong2,sig_vert,delta_mass_KK,cos_PiDs,cos_PiKPhi_3;
  char cand_type;

  fTreeDs->SetBranchAddress("inv_mass",&inv_mass);
  fTreeDs->SetBranchAddress("pt_cand",&pt_cand);
  fTreeDs->SetBranchAddress("d_len",&d_len);
  fTreeDs->SetBranchAddress("d_len_xy",&d_len_xy);
  fTreeDs->SetBranchAddress("norm_dl_xy",&norm_dl_xy);
  fTreeDs->SetBranchAddress("cos_p",&cos_p);
  fTreeDs->SetBranchAddress("cos_p_xy",&cos_p_xy);
  fTreeDs->SetBranchAddress("imp_par_xy",&imp_par_xy);
  fTreeDs->SetBranchAddress("pt_prong0",&pt_prong0);
  fTreeDs->SetBranchAddress("pt_prong1",&pt_prong1);
  fTreeDs->SetBranchAddress("pt_prong2",&pt_prong2);
  fTreeDs->SetBranchAddress("sig_vert",&sig_vert);
  fTreeDs->SetBranchAddress("delta_mass_KK",&delta_mass_KK);
  fTreeDs->SetBranchAddress("cos_PiDs",&cos_PiDs);
  fTreeDs->SetBranchAddress("cos_PiKPhi_3",&cos_PiKPhi_3);
  fTreeDs->SetBranchAddress("cand_type",&cand_type);

  TFile *fout = new TFile(foutput.Data(),"recreate"); 
  TDirectory *dirc = fout->mkdir("PWGHF_TreeCreator");
  dirc->cd();
  TList *listc= new TList(); 
  dirc->Add(listc);
  listc->SetOwner();
  listc->SetName("coutputTreeHFTreeCreator");
  TTree* fTreeDsData = new TTree("fTreeDsData","fTreeDsData");
  listc->Add(fTreeDsData);

  float inv_mass_ML,pt_cand_ML,d_len_ML,d_len_xy_ML,norm_dl_xy_ML,cos_p_ML,cos_p_xy_ML,imp_par_ML,imp_par_xy_ML,pt_prong0_ML,pt_prong1_ML,pt_prong2_ML,sig_vert_ML,delta_mass_KK_ML,cos_PiDs_ML,cos_PiKPhi_3_ML;
  float signal_ML,selected_std_ML;
  char cand_type_ML;

  fTreeDsData->Branch("inv_mass_ML",&inv_mass_ML,"inv_mass_ML/F");
  fTreeDsData->Branch("pt_cand_ML",&pt_cand_ML,"pt_cand_ML/F");
  fTreeDsData->Branch("d_len_ML",&d_len_ML,"d_len_ML/F");
  fTreeDsData->Branch("d_len_xy_ML",&d_len_xy_ML,"d_len_xy_ML/F");
  fTreeDsData->Branch("norm_dl_xy_ML",&norm_dl_xy_ML,"norm_dl_xy_ML/F");
  fTreeDsData->Branch("cos_p_ML",&cos_p_ML,"cos_p_ML/F");
  fTreeDsData->Branch("cos_p_xy_ML",&cos_p_xy_ML,"cos_p_xy_ML/F");
  fTreeDsData->Branch("imp_par_ML",&imp_par_ML,"imp_par_ML/F");
  fTreeDsData->Branch("imp_par_xy_ML",&imp_par_xy_ML,"imp_par_xy_ML/F");
  fTreeDsData->Branch("pt_prong0_ML",&pt_prong0_ML,"pt_prong0_ML/F");
  fTreeDsData->Branch("pt_prong1_ML",&pt_prong1_ML,"pt_prong1_ML/F");
  fTreeDsData->Branch("pt_prong2_ML",&pt_prong2_ML,"pt_prong2_ML/F");
  fTreeDsData->Branch("sig_vert_ML",&sig_vert_ML,"sig_vert_ML/F");
  fTreeDsData->Branch("delta_mass_KK_ML",&delta_mass_KK_ML,"delta_mass_KK_ML/F");
  fTreeDsData->Branch("cos_PiDs_ML",&cos_PiDs_ML,"cos_PiDs_ML/F");
  fTreeDsData->Branch("cos_PiKPhi_3_ML",&cos_PiKPhi_3_ML,"cos_PiKPhi_3_ML/F");
  fTreeDsData->Branch("selected_std_ML",&selected_std_ML,"selected_std_ML/F");
  fTreeDsData->Branch("cand_type_ML",&cand_type_ML);

   
  Long64_t nentries = fTreeDs->GetEntries();
  if (maxevents>nentries || maxevents==-1) maxevents=nentries;
  cout<<" -- Event reading"<<endl;
  for(Long64_t i=0;i<maxevents;i++){ 
    fTreeDs->GetEntry(i);
    if(i%1000==0) cout<<i<<endl;

    selected_std_ML=((int)cand_type&0x1);
    inv_mass_ML=inv_mass;
    pt_cand_ML=pt_cand;
    d_len_ML=d_len;
    d_len_xy_ML=d_len_xy;
    norm_dl_xy_ML=norm_dl_xy;
    cos_p_ML=cos_p;
    cos_p_xy_ML=cos_p_xy;
    imp_par_ML=imp_par;
    imp_par_xy_ML=imp_par_xy;
    pt_prong0_ML=pt_prong0;
    pt_prong1_ML=pt_prong1;
    pt_prong2_ML=pt_prong2;
    sig_vert_ML=sig_vert;
    delta_mass_KK_ML=delta_mass_KK;
    cos_PiDs_ML=cos_PiDs;
    cos_PiKPhi_3_ML=cos_PiKPhi_3;
    cand_type_ML=cand_type;
    fTreeDsData->Fill();
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
    convertTreeROOT6Loop(argv[1],argv[2],argv[3],atoi(argv[4]));
  return 0;
}
