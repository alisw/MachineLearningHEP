#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
using namespace std;
#include <iostream>

bool skimTreePID(TString finput,TString foutput,TString treename, int maxevents=-1){

  // build the signalsample ntuple
  TFile *fin = new TFile(finput.Data()); 
  TTree *fTree = (TTree*)fin->Get(treename.Data()); 
  float pdau0,dedx0,tof0,dca0,sigdca0,chisq0;
  int pdg0,itscl0,tpccl0;

  fTree->SetBranchAddress("pdau0",&pdau0);
  fTree->SetBranchAddress("dedx0",&dedx0);
  fTree->SetBranchAddress("tof0",&tof0);
  fTree->SetBranchAddress("dca0",&dca0);
  fTree->SetBranchAddress("sigdca0",&sigdca0);
  fTree->SetBranchAddress("chisq0",&chisq0);
  fTree->SetBranchAddress("pdg0",&pdg0);
  fTree->SetBranchAddress("itscl0",&itscl0);
  fTree->SetBranchAddress("tpccl0",&tpccl0);

  TFile *fout = new TFile(foutput.Data(),"recreate"); 
  TTree* fTreeML = new TTree("fTreePIDFlagged","fTreePIDFlagged");
  
  float pdau0_ML,dedx0_ML,tof0_ML,dca0_ML,sigdca0_ML,chisq0_ML;
  int pdg0_ML,itscl0_ML,tpccl0_ML;
  
  fTreeML->Branch("pdau0_ML",&pdau0_ML,"pdau0_ML/F");
  fTreeML->Branch("dedx0_ML",&dedx0_ML,"dedx0_ML/F");
  fTreeML->Branch("tof0_ML",&tof0_ML,"tof0_ML/F");
  fTreeML->Branch("dca0_ML",&dca0_ML,"dca0_ML/F");
  fTreeML->Branch("sigdca0_ML",&sigdca0_ML,"sigdca0_ML/F");
  fTreeML->Branch("chisq0_ML",&chisq0_ML,"chisq0_ML/F");
  fTreeML->Branch("pdg0_ML",&pdg0_ML,"pdg0_ML/I");
  fTreeML->Branch("itscl0_ML",&itscl0_ML,"itscl0_ML/I");
  fTreeML->Branch("tpccl0_ML",&tpccl0_ML,"tpccl0_ML/I");
   
  Long64_t nentries = fTree->GetEntries();
  if (maxevents>nentries) {
    std::cout<<"the number of events in the ntupla is smaller than the n. events you want to run on"<<std::endl;
    std::cout<<"we will run on what you have"<<std::endl;
    maxevents=nentries;
  }
  if (maxevents==-1) maxevents=nentries;

  cout<<" -- Event reading"<<endl;
  for(Long64_t i=0;i<maxevents;i++){ 
    fTree->GetEntry(i);
    if(i%1000==0) cout<<i<<endl;
    pdau0_ML=pdau0;
    dedx0_ML=dedx0;
    tof0_ML=tof0;
    dca0_ML=dca0;
    sigdca0_ML=sigdca0;
    chisq0_ML=chisq0;
    pdg0_ML=pdg0;
    itscl0_ML=itscl0;
    tpccl0_ML=tpccl0;
    fTreeML->Fill();
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
    skimTreePID(argv[1],argv[2],argv[3],atoi(argv[4]));
  return 0;
}
