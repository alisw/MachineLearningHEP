//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Oct 26 21:45:28 2018 by ROOT version 6.06/06
// from TTree tree_Event/tree_Event
// found on file: Memory Directory
//////////////////////////////////////////////////////////

#ifndef tree_Event_h
#define tree_Event_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "vector"

using namespace std;

class tree_Event {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   float           centrality;
   float           z_vtx_reco;
   int             n_vtx_contributors;
   int             n_tracks;
   int             is_ev_rej;
   int             run_number;
   float           z_vtx_gen;

   // List of branches
   TBranch        *b_centrality;   //!
   TBranch        *b_z_vtx_reco;   //!
   TBranch        *b_n_vtx_contributors;   //!
   TBranch        *b_n_tracks;   //!
   TBranch        *b_is_ev_rej;   //!
   TBranch        *b_run_number;   //!
   TBranch        *b_z_vtx_gen;   //!

   tree_Event(TTree *tree=0, Bool_t isMC = kFALSE);
   virtual ~tree_Event();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Int_t    GetEntriesFast();
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree, Bool_t isMC);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef tree_Event_cxx
tree_Event::tree_Event(TTree *tree, Bool_t isMC) : fChain(0)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
//   if (tree == 0) {
      //      if (!f || !f->IsOpen()) {
      //   f = new TFile("Memory Directory");
      // }
      //f->GetObject("tree_Event",tree);

      //}
   Init(tree, isMC);
}

tree_Event::~tree_Event()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t tree_Event::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Int_t tree_Event::GetEntriesFast()
{
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntriesFast();
}
Long64_t tree_Event::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void tree_Event::Init(TTree *tree, Bool_t isMC)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   centrality = 0;
   z_vtx_reco = 0;
   n_vtx_contributors = 0;
   n_tracks = 0;
   is_ev_rej = 0;
   run_number = 0;
   z_vtx_gen = 0;
    
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("centrality", &centrality, &b_centrality);
   fChain->SetBranchAddress("z_vtx_reco", &z_vtx_reco, &b_z_vtx_reco);
   fChain->SetBranchAddress("n_vtx_contributors", &n_vtx_contributors, &b_n_vtx_contributors);
   fChain->SetBranchAddress("n_tracks", &n_tracks, &b_n_tracks);
   fChain->SetBranchAddress("is_ev_rej", &is_ev_rej, &b_is_ev_rej);
   fChain->SetBranchAddress("run_number", &run_number, &b_run_number);
   if(isMC) fChain->SetBranchAddress("z_vtx_gen", &z_vtx_gen, &b_z_vtx_gen);

   Notify();
}

Bool_t tree_Event::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void tree_Event::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t tree_Event::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef tree_Event_cxx
