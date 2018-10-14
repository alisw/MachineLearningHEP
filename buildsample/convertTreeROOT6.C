#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
using namespace std;
#include <iostream>

void convertTreeROOT6(TString finput,TString treename,TString output,int maxevents){

  // build the signalsample ntuple
  TFile *fin = new TFile(finput.Data()); 
  TTree *fTreeDs = (TTree*)fin->Get(treename.Data()); 

  TFile* outf = TFile::Open(output,"recreate");
  TTree* nfTreeDs = fTreeDs->CloneTree(0);

  Long64_t nentries = fTreeDs->GetEntries();

  cout<<" -- Event reading"<<endl;
  for(Long64_t i=0;i<maxevents;i++){
    fTreeDs->GetEntry(i);
    nfTreeDs->Fill();
  }
  outf->Write();
  cout<<" -- Writing new trees done"<<endl;
  outf->Close(); 
}

int main(int argc, char *argv[])
{
  if((argc != 5))
  {
    std::cout << "Wrong number of inputs" << std::endl;
    return 1;
  }
  
  if(argc == 5)
    convertTreeROOT6(argv[1],argv[2],argv[3],atoi(argv[4]));
  return 0;
}
