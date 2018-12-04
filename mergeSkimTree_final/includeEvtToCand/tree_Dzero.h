//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Oct 26 21:45:28 2018 by ROOT version 6.06/06
// from TTree tree_Dzero/tree_Dzero
// found on file: Memory Directory
//////////////////////////////////////////////////////////

#ifndef tree_Dzero_h
#define tree_Dzero_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "vector"

using namespace std;

class tree_Dzero {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   UInt_t          n_cand;
   vector<int>     *cand_type;
   vector<float>   *inv_mass;
   vector<float>   *pt_cand;
   vector<float>   *y_cand;
   vector<float>   *eta_cand;
   vector<float>   *phi_cand;
   vector<float>   *d_len;
   vector<float>   *d_len_xy;
   vector<float>   *norm_dl_xy;
   vector<float>   *cos_p;
   vector<float>   *cos_p_xy;
   vector<float>   *imp_par_xy;
   vector<float>   *max_norm_d0d0exp;
   vector<float>   *p_prong0;
   vector<float>   *pt_prong0;
   vector<float>   *eta_prong0;
   vector<float>   *imp_par_prong0;
   vector<float>   *phi_prong0;
   vector<float>   *p_prong1;
   vector<float>   *pt_prong1;
   vector<float>   *eta_prong1;
   vector<float>   *imp_par_prong1;
   vector<float>   *phi_prong1;
   vector<float>   *p_prong2;
   vector<float>   *pt_prong2;
   vector<float>   *eta_prong2;
   vector<float>   *imp_par_prong2;
   vector<float>   *phi_prong2;

   vector<int>     *nTPCcls_prong0;
   vector<int>     *nTPCclspid_prong0;
   vector<float>   *nTPCcrossrow_prong0;
   vector<float>   *chi2perndf_prong0;
   vector<int>     *nITScls_prong0;
   vector<int>     *ITSclsmap_prong0;
   vector<int>     *nTPCcls_prong1;
   vector<int>     *nTPCclspid_prong1;
   vector<float>   *nTPCcrossrow_prong1;
   vector<float>   *chi2perndf_prong1;
   vector<int>     *nITScls_prong1;
   vector<int>     *ITSclsmap_prong1;
   vector<int>     *nTPCcls_prong2;
   vector<int>     *nTPCclspid_prong2;
   vector<float>   *nTPCcrossrow_prong2;
   vector<float>   *chi2perndf_prong2;
   vector<int>     *nITScls_prong2;
   vector<int>     *ITSclsmap_prong2;

   vector<int>     *nsigTPC_Pi_0;
   vector<int>     *nsigTPC_K_0;
   vector<int>     *nsigTOF_Pi_0;
   vector<int>     *nsigTOF_K_0;
   vector<float>   *dEdxTPC_0;
   vector<float>   *ToF_0;
   vector<float>   *pTPC_prong0;
   vector<float>   *pTOF_prong0;
   vector<float>   *trlen_prong0;
   vector<float>   *start_time_res_prong0;
   vector<int>     *nsigTPC_Pi_1;
   vector<int>     *nsigTPC_K_1;
   vector<int>     *nsigTOF_Pi_1;
   vector<int>     *nsigTOF_K_1;
   vector<float>   *dEdxTPC_1;
   vector<float>   *ToF_1;
   vector<float>   *pTPC_prong1;
   vector<float>   *pTOF_prong1;
   vector<float>   *trlen_prong1;
   vector<float>   *start_time_res_prong1;
   vector<int>     *nsigTPC_Pi_2;
   vector<int>     *nsigTPC_K_2;
   vector<int>     *nsigTOF_Pi_2;
   vector<int>     *nsigTOF_K_2;
   vector<float>   *dEdxTPC_2;
   vector<float>   *ToF_2;
   vector<float>   *pTPC_prong2;
   vector<float>   *pTOF_prong2;
   vector<float>   *trlen_prong2;
   vector<float>   *start_time_res_prong2;
    
   // List of branches
   TBranch        *b_n_cand;   //!
   TBranch        *b_cand_type;   //!
   TBranch        *b_inv_mass;   //!
   TBranch        *b_pt_cand;   //!
   TBranch        *b_y_cand;   //!
   TBranch        *b_eta_cand;   //!
   TBranch        *b_phi_cand;   //!
   TBranch        *b_d_len;   //!
   TBranch        *b_d_len_xy;   //!
   TBranch        *b_norm_dl_xy;   //!
   TBranch        *b_cos_p;   //!
   TBranch        *b_cos_p_xy;   //!
   TBranch        *b_imp_par_xy;   //!
   TBranch        *b_max_norm_d0d0exp;   //!
   TBranch        *b_p_prong0;   //!
   TBranch        *b_pt_prong0;   //!
   TBranch        *b_eta_prong0;   //!
   TBranch        *b_imp_par_prong0;   //!
   TBranch        *b_phi_prong0;   //!
   TBranch        *b_p_prong1;   //!
   TBranch        *b_pt_prong1;   //!
   TBranch        *b_eta_prong1;   //!
   TBranch        *b_imp_par_prong1;   //!
   TBranch        *b_phi_prong1;   //!
   TBranch        *b_p_prong2;   //!
   TBranch        *b_pt_prong2;   //!
   TBranch        *b_eta_prong2;   //!
   TBranch        *b_imp_par_prong2;   //!
   TBranch        *b_phi_prong2;   //!

   TBranch        *b_nTPCcls_prong0;   //!
   TBranch        *b_nTPCclspid_prong0;   //!
   TBranch        *b_nTPCcrossrow_prong0;   //!
   TBranch        *b_chi2perndf_prong0;   //!
   TBranch        *b_nITScls_prong0;   //!
   TBranch        *b_ITSclsmap_prong0;   //!
   TBranch        *b_nTPCcls_prong1;   //!
   TBranch        *b_nTPCclspid_prong1;   //!
   TBranch        *b_nTPCcrossrow_prong1;   //!
   TBranch        *b_chi2perndf_prong1;   //!
   TBranch        *b_nITScls_prong1;   //!
   TBranch        *b_ITSclsmap_prong1;   //!
   TBranch        *b_nTPCcls_prong2;   //!
   TBranch        *b_nTPCclspid_prong2;   //!
   TBranch        *b_nTPCcrossrow_prong2;   //!
   TBranch        *b_chi2perndf_prong2;   //!
   TBranch        *b_nITScls_prong2;   //!
   TBranch        *b_ITSclsmap_prong2;   //!

   TBranch        *b_nsigTPC_Pi_0;   //!
   TBranch        *b_nsigTPC_K_0;   //!
   TBranch        *b_nsigTOF_Pi_0;   //!
   TBranch        *b_nsigTOF_K_0;   //!
   TBranch        *b_dEdxTPC_0;   //!
   TBranch        *b_ToF_0;   //!
   TBranch        *b_pTPC_prong0;   //!
   TBranch        *b_pTOF_prong0;   //!
   TBranch        *b_trlen_prong0;   //!
   TBranch        *b_start_time_res_prong0;   //!
   TBranch        *b_nsigTPC_Pi_1;   //!
   TBranch        *b_nsigTPC_K_1;   //!
   TBranch        *b_nsigTOF_Pi_1;   //!
   TBranch        *b_nsigTOF_K_1;   //!
   TBranch        *b_dEdxTPC_1;   //!
   TBranch        *b_ToF_1;   //!
   TBranch        *b_pTPC_prong1;   //!
   TBranch        *b_pTOF_prong1;   //!
   TBranch        *b_trlen_prong1;   //!
   TBranch        *b_start_time_res_prong1;   //!
   TBranch        *b_nsigTPC_Pi_2;   //!
   TBranch        *b_nsigTPC_K_2;   //!
   TBranch        *b_nsigTOF_Pi_2;   //!
   TBranch        *b_nsigTOF_K_2;   //!
   TBranch        *b_dEdxTPC_2;   //!
   TBranch        *b_ToF_2;   //!
   TBranch        *b_pTPC_prong2;   //!
   TBranch        *b_pTOF_prong2;   //!
   TBranch        *b_trlen_prong2;   //!
   TBranch        *b_start_time_res_prong2;   //!

   tree_Dzero(TTree *tree=0);
   virtual ~tree_Dzero();
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

#ifdef tree_Dzero_cxx
tree_Dzero::tree_Dzero(TTree *tree) : fChain(0)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
//   if (tree == 0) {
      //      if (!f || !f->IsOpen()) {
      //   f = new TFile("Memory Directory");
      // }
      //f->GetObject("tree_Dzero",tree);

      //}
   Init(tree);
}

tree_Dzero::~tree_Dzero()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t tree_Dzero::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Int_t tree_Dzero::GetEntriesFast()
{
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntriesFast();
}
Long64_t tree_Dzero::LoadTree(Long64_t entry)
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

void tree_Dzero::Init(TTree *tree)
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
   inv_mass = 0;
   pt_cand = 0;
   y_cand = 0;
   eta_cand = 0;
   phi_cand = 0;
   d_len = 0;
   d_len_xy = 0;
   norm_dl_xy = 0;
   cos_p = 0;
   cos_p_xy = 0;
   imp_par_xy = 0;
   max_norm_d0d0exp = 0;
   p_prong0 = 0;
   pt_prong0 = 0;
   eta_prong0 = 0;
   imp_par_prong0 = 0;
   phi_prong0 = 0;
   p_prong1 = 0;
   pt_prong1 = 0;
   eta_prong1 = 0;
   imp_par_prong1 = 0;
   phi_prong1 = 0;
   p_prong2 = 0;
   pt_prong2 = 0;
   eta_prong2 = 0;
   imp_par_prong2 = 0;
   phi_prong2 = 0;

   nTPCcls_prong0 = 0;
   nTPCclspid_prong0 = 0;
   nTPCcrossrow_prong0 = 0;
   chi2perndf_prong0 = 0;
   nITScls_prong0 = 0;
   ITSclsmap_prong0 = 0;
   nTPCcls_prong1 = 0;
   nTPCclspid_prong1 = 0;
   nTPCcrossrow_prong1 = 0;
   chi2perndf_prong1 = 0;
   nITScls_prong1 = 0;
   ITSclsmap_prong1 = 0;
   nTPCcls_prong2 = 0;
   nTPCclspid_prong2 = 0;
   nTPCcrossrow_prong2 = 0;
   chi2perndf_prong2 = 0;
   nITScls_prong2 = 0;
   ITSclsmap_prong2 = 0;

   nsigTPC_Pi_0 = 0;
   nsigTPC_K_0 = 0;
   nsigTOF_Pi_0 = 0;
   nsigTOF_K_0 = 0;
   dEdxTPC_0 = 0;
   ToF_0 = 0;
   pTPC_prong0 = 0;
   pTOF_prong0 = 0;
   trlen_prong0 = 0;
   start_time_res_prong0 = 0;
   nsigTPC_Pi_1 = 0;
   nsigTPC_K_1 = 0;
   nsigTOF_Pi_1 = 0;
   nsigTOF_K_1 = 0;
   dEdxTPC_1 = 0;
   ToF_1 = 0;
   pTPC_prong1 = 0;
   pTOF_prong1 = 0;
   trlen_prong1 = 0;
   start_time_res_prong1 = 0;
   nsigTPC_Pi_2 = 0;
   nsigTPC_K_2 = 0;
   nsigTOF_Pi_2 = 0;
   nsigTOF_K_2 = 0;
   dEdxTPC_2 = 0;
   ToF_2 = 0;
   pTPC_prong2 = 0;
   pTOF_prong2 = 0;
   trlen_prong2 = 0;
   start_time_res_prong2 = 0;
    
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("n_cand", &n_cand, &b_n_cand);
   fChain->SetBranchAddress("cand_type", &cand_type, &b_cand_type);
   fChain->SetBranchAddress("inv_mass", &inv_mass, &b_inv_mass);
   fChain->SetBranchAddress("pt_cand", &pt_cand, &b_pt_cand);
   fChain->SetBranchAddress("y_cand", &y_cand, &b_y_cand);
   fChain->SetBranchAddress("eta_cand", &eta_cand, &b_eta_cand);
   fChain->SetBranchAddress("phi_cand", &phi_cand, &b_phi_cand);
   fChain->SetBranchAddress("d_len", &d_len, &b_d_len);
   fChain->SetBranchAddress("d_len_xy", &d_len_xy, &b_d_len_xy);
   fChain->SetBranchAddress("norm_dl_xy", &norm_dl_xy, &b_norm_dl_xy);
   fChain->SetBranchAddress("cos_p", &cos_p, &b_cos_p);
   fChain->SetBranchAddress("cos_p_xy", &cos_p_xy, &b_cos_p_xy);
   fChain->SetBranchAddress("imp_par_xy", &imp_par_xy, &b_imp_par_xy);
   fChain->SetBranchAddress("max_norm_d0d0exp", &max_norm_d0d0exp, &b_max_norm_d0d0exp);
   fChain->SetBranchAddress("p_prong0", &p_prong0, &b_p_prong0);
   fChain->SetBranchAddress("pt_prong0", &pt_prong0, &b_pt_prong0);
   fChain->SetBranchAddress("eta_prong0", &eta_prong0, &b_eta_prong0);
   fChain->SetBranchAddress("imp_par_prong0", &imp_par_prong0, &b_imp_par_prong0);
   fChain->SetBranchAddress("phi_prong0", &phi_prong0, &b_phi_prong0);
   fChain->SetBranchAddress("p_prong1", &p_prong1, &b_p_prong1);
   fChain->SetBranchAddress("pt_prong1", &pt_prong1, &b_pt_prong1);
   fChain->SetBranchAddress("eta_prong1", &eta_prong1, &b_eta_prong1);
   fChain->SetBranchAddress("imp_par_prong1", &imp_par_prong1, &b_imp_par_prong1);
   fChain->SetBranchAddress("phi_prong1", &phi_prong1, &b_phi_prong1);
   fChain->SetBranchAddress("p_prong2", &p_prong2, &b_p_prong2);
   fChain->SetBranchAddress("pt_prong2", &pt_prong2, &b_pt_prong2);
   fChain->SetBranchAddress("eta_prong2", &eta_prong2, &b_eta_prong2);
   fChain->SetBranchAddress("imp_par_prong2", &imp_par_prong2, &b_imp_par_prong2);
   fChain->SetBranchAddress("phi_prong2", &phi_prong2, &b_phi_prong2);
    
   fChain->SetBranchAddress("nTPCcls_prong0", &nTPCcls_prong0, &b_nTPCcls_prong0);
   fChain->SetBranchAddress("nTPCclspid_prong0", &nTPCclspid_prong0, &b_nTPCclspid_prong0);
   fChain->SetBranchAddress("nTPCcrossrow_prong0", &nTPCcrossrow_prong0, &b_nTPCcrossrow_prong0);
   fChain->SetBranchAddress("chi2perndf_prong0", &chi2perndf_prong0, &b_nTPCcrossrow_prong0);
   fChain->SetBranchAddress("nITScls_prong0", &nITScls_prong0, &b_nITScls_prong0);
   fChain->SetBranchAddress("ITSclsmap_prong0", &ITSclsmap_prong0, &b_ITSclsmap_prong0);
   fChain->SetBranchAddress("nTPCcls_prong1", &nTPCcls_prong1, &b_nTPCcls_prong1);
   fChain->SetBranchAddress("nTPCclspid_prong1", &nTPCclspid_prong1, &b_nTPCclspid_prong1);
   fChain->SetBranchAddress("nTPCcrossrow_prong1", &nTPCcrossrow_prong1, &b_nTPCcrossrow_prong1);
   fChain->SetBranchAddress("chi2perndf_prong1", &chi2perndf_prong1, &b_chi2perndf_prong1);
   fChain->SetBranchAddress("nITScls_prong1", &nITScls_prong1, &b_nITScls_prong1);
   fChain->SetBranchAddress("ITSclsmap_prong1", &ITSclsmap_prong1, &b_ITSclsmap_prong1);
   fChain->SetBranchAddress("nTPCcls_prong2", &nTPCcls_prong2, &b_nTPCcls_prong2);
   fChain->SetBranchAddress("nTPCclspid_prong2", &nTPCclspid_prong2, &b_nTPCclspid_prong2);
   fChain->SetBranchAddress("nTPCcrossrow_prong2", &nTPCcrossrow_prong2, &b_nTPCcrossrow_prong2);
   fChain->SetBranchAddress("chi2perndf_prong2", &chi2perndf_prong2, &b_chi2perndf_prong2);
   fChain->SetBranchAddress("nITScls_prong2", &nITScls_prong2, &b_nITScls_prong2);
   fChain->SetBranchAddress("ITSclsmap_prong2", &ITSclsmap_prong2, &b_ITSclsmap_prong2);

   fChain->SetBranchAddress("nsigTPC_Pi_0", &nsigTPC_Pi_0, &b_nsigTPC_Pi_0);
   fChain->SetBranchAddress("nsigTPC_K_0", &nsigTPC_K_0, &b_nsigTPC_K_0);
   fChain->SetBranchAddress("nsigTOF_Pi_0", &nsigTOF_Pi_0, &b_nsigTOF_Pi_0);
   fChain->SetBranchAddress("nsigTOF_K_0", &nsigTOF_K_0, &b_nsigTOF_K_0);
   fChain->SetBranchAddress("dEdxTPC_0", &dEdxTPC_0, &b_dEdxTPC_0);
   fChain->SetBranchAddress("ToF_0", &ToF_0, &b_ToF_0);
   fChain->SetBranchAddress("pTPC_prong0", &pTPC_prong0, &b_pTPC_prong0);
   fChain->SetBranchAddress("pTOF_prong0", &pTOF_prong0, &b_pTOF_prong0);
   fChain->SetBranchAddress("trlen_prong0", &trlen_prong0, &b_trlen_prong0);
   fChain->SetBranchAddress("start_time_res_prong0", &start_time_res_prong0, &b_start_time_res_prong0);
    fChain->SetBranchAddress("nsigTPC_Pi_1", &nsigTPC_Pi_1, &b_nsigTPC_Pi_1);
    fChain->SetBranchAddress("nsigTPC_K_1", &nsigTPC_K_1, &b_nsigTPC_K_1);
    fChain->SetBranchAddress("nsigTOF_Pi_1", &nsigTOF_Pi_1, &b_nsigTOF_Pi_1);
    fChain->SetBranchAddress("nsigTOF_K_1", &nsigTOF_K_1, &b_nsigTOF_K_1);
    fChain->SetBranchAddress("dEdxTPC_1", &dEdxTPC_1, &b_dEdxTPC_1);
    fChain->SetBranchAddress("ToF_1", &ToF_1, &b_ToF_1);
    fChain->SetBranchAddress("pTPC_prong1", &pTPC_prong1, &b_pTPC_prong1);
    fChain->SetBranchAddress("pTOF_prong1", &pTOF_prong1, &b_pTOF_prong1);
    fChain->SetBranchAddress("trlen_prong1", &trlen_prong1, &b_trlen_prong1);
    fChain->SetBranchAddress("start_time_res_prong1", &start_time_res_prong1, &b_start_time_res_prong1);
   fChain->SetBranchAddress("nsigTPC_Pi_2", &nsigTPC_Pi_2, &b_nsigTPC_Pi_2);
   fChain->SetBranchAddress("nsigTPC_K_2", &nsigTPC_K_2, &b_nsigTPC_K_2);
   fChain->SetBranchAddress("nsigTOF_Pi_2", &nsigTOF_Pi_2, &b_nsigTOF_Pi_2);
   fChain->SetBranchAddress("nsigTOF_K_2", &nsigTOF_K_2, &b_nsigTOF_K_2);
   fChain->SetBranchAddress("dEdxTPC_2", &dEdxTPC_2, &b_dEdxTPC_2);
   fChain->SetBranchAddress("ToF_2", &ToF_2, &b_ToF_2);
   fChain->SetBranchAddress("pTPC_prong2", &pTPC_prong2, &b_pTPC_prong2);
   fChain->SetBranchAddress("pTOF_prong2", &pTOF_prong2, &b_pTOF_prong2);
   fChain->SetBranchAddress("trlen_prong2", &trlen_prong2, &b_trlen_prong2);
   fChain->SetBranchAddress("start_time_res_prong2", &start_time_res_prong2, &b_start_time_res_prong2);
   Notify();
}

Bool_t tree_Dzero::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void tree_Dzero::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t tree_Dzero::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef tree_Dzero_cxx
