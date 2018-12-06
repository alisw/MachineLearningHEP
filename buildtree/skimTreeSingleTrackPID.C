#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
using namespace std;
#include <iostream>

bool skimTreeSingleTrackPID(TString finput,TString foutput,TString treename, int maxevents=-1){

  cout<<" -- Checking if input and output files are same"<<endl;
  if(finput==foutput)
    {
      cout<<"    Error: Input file will be overwritten."<<endl;
      return 0;
    }
  TFile* inf = TFile::Open(finput);
  TTree* ntinput = (TTree*)inf->Get(treename);

  TFile* fout = TFile::Open(foutput,"recreate");
  TTree* ntinputsel = ntinput->CloneTree(0);

  Long64_t nentries = ntinput->GetEntries();
  std::cout<<"number of entries"<<nentries<<std::endl;
  std::cout<<nentries<<std::endl;
  if (maxevents>nentries) {
    std::cout<<"the number of events in the ntupla is smaller than the n. events you want to run on"<<std::endl;
    std::cout<<"we will run on what you have"<<std::endl;
    maxevents=nentries;
  }
  if (maxevents==-1) maxevents=nentries;

  for(Long64_t i=0;i<maxevents;i++)
    {
      if(i%1000==0) cout<<i<<endl;
      ntinput->GetEntry(i);

      ntinputsel->Fill();
    }
  fout->Write();
  cout<<" -- Writing new trees done"<<endl;
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
    skimTreeSingleTrackPID(argv[1],argv[2],argv[3],atoi(argv[4]));
  return 0;
}

