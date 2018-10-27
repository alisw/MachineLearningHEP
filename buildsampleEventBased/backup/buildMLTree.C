#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
using namespace std;
double fMassDs=1.967;
double fMassDsRange=0.05;
#include <iostream>

bool makeMLTree(TString finput,TString foutput,TString treename,int signalsample=1,int maxevents=1000){

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
  
  Long64_t nentries = fTreeDs->GetEntries();
  if (maxevents>nentries) {
    std::cout<<"the number of events in the ntupla is smaller than the n. events you want to run on"<<std::endl;
    std::cout<<"this skimming is going to fail, so sad. "<<std::endl;
    return false;
  }
  cout<<" -- Event reading"<<endl;
  for(Long64_t i=0;i<maxevents;i++){ 
    fTreeDs->GetEntry(i);
    if(i%1000==0) cout<<i<<endl;
    
    bool issignalPT=((cand_type>>3)&0x1)&&((cand_type>>1)&0x1);
    bool issignalFD=((cand_type>>4)&0x1)&&((cand_type>>1)&0x1);
    if ((int)cand_type>0) std::cout<<"-------------------------------------------------------------------"<<std::endl;
    
    if (signalsample==1){
      if (!(issignalPT==1 || issignalFD==1)) continue; 
    }
    if (signalsample==0){
      if (fabs(inv_mass-fMassDs)<fMassDsRange) continue; 
    }
    selected_std_ML=(cand_type&0x1);
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
    signal_ML=signalsample;
    fTreeDsML->Fill();
    } 
  fout->Write();
  fout->Close();
  return true; 
  
  
}

void buildMLTree(TString MCsample, TString MCtree, int nsignals, TString Datasample, TString Datatree, int nbkg ){
  bool flagMC=makeMLTree(MCsample.Data(),Form("treeSignalN%d.root",nsignals),MCtree.Data(),1,nsignals);
  bool flagData=makeMLTree(Datasample.Data(),Form("treeBkgN%dPreMassCut.root",nbkg),Datatree.Data(),0,nbkg);
  if (!(flagMC==true&&flagData==true)) return;
  TChain ch("fTreeDsFlagged");
  ch.Add(Form("treeSignalN%d.root",nsignals));
  ch.Add(Form("treeBkgN%dPreMassCut.root",nbkg));
  ch.Merge(Form("treeTotalSignalN%dBkgN%dPreMassCut.root",nsignals,nbkg));

}

int main(int argc, char *argv[])
{
  if((argc != 7))
  {
    std::cout << "Wrong number of inputs" << std::endl;
    return 1;
  }
  
  if(argc == 7)
    buildMLTree(argv[1],argv[2],atoi(argv[3]),argv[4],argv[5],atoi(argv[6]));
  return 0;
}
