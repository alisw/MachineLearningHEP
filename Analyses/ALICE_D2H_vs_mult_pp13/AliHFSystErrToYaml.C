
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using std::cout;
using std::endl;

#include "TCanvas.h"
#include "TDirectoryFile.h"
#include "TF1.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "THnSparse.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TList.h"
#include "TPad.h"
#include "TROOT.h"
#include "TSystem.h"
#include "AliHFSystErr.h"


void AliHFSystErrToYaml(TString inputrootfile, TString outputyamlfile, Bool_t isLc = kFALSE){
  
  /*
  const Int_t nPtBins = 5;
  Float_t PtBins[nPtBins+1] = {2., 4., 6., 8., 12., 24.};
  Float_t midPtBins[nPtBins] = {3., 5., 7., 10., 18.};
  */
  const Int_t nPtBins = 6;
  Float_t PtBins[nPtBins+1] = {1., 2., 4., 6., 8., 12., 24.};
  Float_t midPtBins[nPtBins] = {1.5, 2.9, 5., 7., 10., 18.};
  
  TFile* fInput =  new TFile(inputrootfile.Data(),"read");
  
  AliHFSystErr* oSystErr = (AliHFSystErr*)fInput->Get("AliHFSystErr");
  
  ofstream fOutYaml (outputyamlfile.Data());
  
  fOutYaml <<endl <<endl;
  fOutYaml << "#Extracted automatically from AliHFSystErrObject of file: " << inputrootfile << endl;
  
  // yield
  fOutYaml <<"names: [\"yield\", \"cut\", \"track\", \"ptshape\", \"feeddown_mult\", \"trigger\", \"multiplicity_weights\", \"multiplicity_interval\", \"branching_ratio\", \"sigmav0\"]" <<endl <<endl;
  
  
  fOutYaml <<"yield:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    fOutYaml <<"    - [0., 0., " <<oSystErr->GetRawYieldErr(midPtBins[ipt]) <<", " <<oSystErr->GetRawYieldErr(midPtBins[ipt]) <<"]" <<endl;
  }
  fOutYaml <<endl;
  
  
  fOutYaml <<"cut:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    //fOutYaml <<"    - [0., 0., " <<oSystErr->GetCutsEffErr(midPtBins[ipt]) <<", " <<oSystErr->GetCutsEffErr(midPtBins[ipt]) <<"]" <<endl;
    fOutYaml <<"    - [0., 0., " << TMath::Sqrt( oSystErr->GetCutsEffErr(midPtBins[ipt]) * oSystErr->GetCutsEffErr(midPtBins[ipt]) + oSystErr->GetPIDEffErr(midPtBins[ipt]) * oSystErr->GetPIDEffErr(midPtBins[ipt])) <<", " << TMath::Sqrt( oSystErr->GetCutsEffErr(midPtBins[ipt]) * oSystErr->GetCutsEffErr(midPtBins[ipt]) + oSystErr->GetPIDEffErr(midPtBins[ipt]) * oSystErr->GetPIDEffErr(midPtBins[ipt]))  <<"]" <<endl;
  }
  fOutYaml <<endl;

  
  /*
  // PID  ---> Included in cut as for ML!
  fOutYaml <<"PID:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    fOutYaml <<"    - [0., 0., " <<oSystErr->GetPIDEffErr(midPtBins[ipt]) <<", " <<oSystErr->GetPIDEffErr(midPtBins[ipt]) <<"]" <<endl;
  }
  fOutYaml <<endl;
  */
  

  fOutYaml <<"track:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    fOutYaml <<"    - [0., 0., " <<oSystErr->GetTrackingEffErr(midPtBins[ipt]) <<", " <<oSystErr->GetTrackingEffErr(midPtBins[ipt]) <<"]" <<endl;
  }
  fOutYaml <<endl;
  

  fOutYaml <<"ptshape:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    fOutYaml <<"    - [0., 0., " <<oSystErr->GetMCPtShapeErr(midPtBins[ipt]) <<", " <<oSystErr->GetMCPtShapeErr(midPtBins[ipt]) <<"]" <<endl;
  }
  fOutYaml <<endl;
  
  
  fOutYaml <<"feeddown_mult:" <<endl;
  //TString fpromptname = "gFcConservative";
  //if(isLc) fpromptname = "gFcCorrConservative";
  //TGraphAsymmErrors* fPrompt = (TGraphAsymmErrors*)fInput->Get(fpromptname.Data());
  for(int ipt=0; ipt<nPtBins; ipt++){
    //double errYhigh = fPrompt->GetErrorYhigh(ipt+1);
    //double errYlow = fPrompt->GetErrorYlow(ipt+1);
    //NOT PART OF ALIHFSYSTERR, TO BE FILLED IN BY HAND
    fOutYaml <<"    - [0., 0., " <<0. <<", " <<0. <<"]" <<endl;
  }
  fOutYaml <<endl;
  

  fOutYaml <<"trigger:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    //NOT PART OF ALIHFSYSTERR, TO BE FILLED IN BY HAND
    fOutYaml <<"    - [0., 0., " <<0. <<", " <<0. <<"]" <<endl;
  }
  fOutYaml <<endl;
  

  fOutYaml <<"multiplicity_weights:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    //NOT PART OF ALIHFSYSTERR, TO BE FILLED IN BY HAND
    fOutYaml <<"    - [0., 0., " <<0. <<", " <<0. <<"]" <<endl;
  }
  fOutYaml <<endl;
  
  
  fOutYaml <<"multiplicity_interval:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    //NOT PART OF ALIHFSYSTERR, TO BE FILLED IN BY HAND
    fOutYaml <<"    - [0., 0., " <<0. <<", " <<0. <<"]" <<endl;
  }
  fOutYaml <<endl;
  
  
  fOutYaml <<"branching_ratio:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    if(!isLc) fOutYaml <<"    - [0., 0., " <<0.0129 <<", " <<0.0129 <<"]" <<endl;
    else fOutYaml <<"    - [0., 0., " <<0.053 <<", " <<0.053 <<"]" <<endl;
    //fOutYaml <<"    - [0., 0., " <<oSystErr->GetBRErr() <<", " <<oSystErr->GetBRErr() <<"]" <<endl;
  }
  fOutYaml <<endl;
  
  
  fOutYaml <<"sigmav0:" <<endl;
  for(int ipt=0; ipt<nPtBins; ipt++){
    fOutYaml <<"    - [0., 0., " <<0.05 <<", " <<0.05 <<"]" <<endl;
    //fOutYaml <<"    - [0., 0., " <<oSystErr->GetNormErr() <<", " <<oSystErr->GetNormErr() <<"]" <<endl;
  }
  fOutYaml <<endl;
  
  
  fOutYaml.close();
}
