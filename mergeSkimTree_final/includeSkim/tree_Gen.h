//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Oct 26 21:45:28 2018 by ROOT version 6.06/06
// from TTree tree_Gen/tree_Gen
// found on file: Memory Directory
//////////////////////////////////////////////////////////

#ifndef tree_Gen_h
#define tree_Gen_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "vector"

using namespace std;

class tree_Gen {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   UInt_t          n_cand;
   vector<int>     *cand_type;
   vector<float>   *pt_cand;
   vector<float>   *y_cand;
   vector<float>   *eta_cand;
   vector<float>   *phi_cand;
   vector<bool>    *dau_in_acc;
    
   // List of branches
   TBranch        *b_n_cand;   //!
   TBranch        *b_cand_type;   //!
   TBranch        *b_pt_cand;   //!
   TBranch        *b_y_cand;   //!
   TBranch        *b_eta_cand;   //!
   TBranch        *b_phi_cand;   //!
   TBranch        *b_dau_in_acc;   //!

   tree_Gen(TTree *tree=0);
   virtual ~tree_Gen();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Int_t    GetEntriesFast();
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef tree_Gen_cxx
tree_Gen::tree_Gen(TTree *tree) : fChain(0)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
//   if (tree == 0) {
      //      if (!f || !f->IsOpen()) {
      //   f = new TFile("Memory Directory");
      // }
      //f->GetObject("tree_Gen",tree);

      //}

   //if not MC, tree will be zero
   if(tree) Init(tree);
}

tree_Gen::~tree_Gen()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t tree_Gen::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Int_t tree_Gen::GetEntriesFast()
{
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntriesFast();
}
Long64_t tree_Gen::LoadTree(Long64_t entry)
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

void tree_Gen::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   cand_type = 0;
   pt_cand = 0;
   y_cand = 0;
   eta_cand = 0;
   phi_cand = 0;
   dau_in_acc = 0;
    
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("n_cand", &n_cand, &b_n_cand);
   fChain->SetBranchAddress("cand_type", &cand_type, &b_cand_type);
   fChain->SetBranchAddress("pt_cand", &pt_cand, &b_pt_cand);
   fChain->SetBranchAddress("y_cand", &y_cand, &b_y_cand);
   fChain->SetBranchAddress("eta_cand", &eta_cand, &b_eta_cand);
   fChain->SetBranchAddress("phi_cand", &phi_cand, &b_phi_cand);
   fChain->SetBranchAddress("dau_in_acc", &dau_in_acc, &b_dau_in_acc);

   Notify();
}

Bool_t tree_Gen::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void tree_Gen::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t tree_Gen::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef tree_Gen_cxx
