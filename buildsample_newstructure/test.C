#include <iostream>
#include <vector>
#include <algorithm>
#include "tree_Ds.C"

using namespace std;

void test(){
  tree_Ds * t = new tree_Ds();
  int nevt = t->GetEntriesFast();
  cout<<nevt<<endl;
}
//for (Long64_t jentry=0; jentry<nevt;jentry++) {
// t -> GetEntry(jentry)

//  std::cout << t->n_cand <<std:endl;
//}
