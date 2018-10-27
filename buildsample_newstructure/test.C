#include <iostream>
#include <vector>
#include <algorithm>
#include "tree_Ds.C"

using namespace std;

void test(){
  tree_Ds t;
  int nevt = t.GetEntriesFast();
  cout << nevt << endl;
  for(Long64_t jentry=0; jentry<nevt;jentry++){
    t.GetEntry(jentry);   
    for(int icand = 0; icand < t.n_cand; icand++){ 
      cout << t.pt_cand -> at(icand) << endl;
    }
  }
}
