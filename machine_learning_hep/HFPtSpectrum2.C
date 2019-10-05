#if !defined(__CINT__) || defined(__MAKECINT__)
#include <Riostream.h>
#include "TH1D.h"
#include "TH1.h"
#include "TH2F.h"
#include "TNtuple.h"
#include "TFile.h"
#include "TSystem.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLegend.h"
#include "AliHFSystErr.h"
#endif

//
// Macro to use mimic AliHFPtSpectrum class using HFPtSpectrum file as input
// used for high multiplicity analyses where fPrompt calculation with Nb 
// method fails due to multiplicity-independent FONLL prediction.
//
// Besides using the fPrompt fraction from the input file, also the syst.
// errors (AliHFSystErr class) is copied.
//
// NB: So far only works for pp, using Nb method.
//

void HFPtSpectrum2 (const char *inputCrossSection, 
                    const char *efffilename="Efficiencies.root",
                    const char *nameeffprompt= "eff",
                    const char *nameefffeed = "effB",
                    const char *recofilename="Reconstructed.root",
                    const char *recohistoname="hRawSpectrumD0",
                    const char *outfilename="HFPtSpectrum.root",
                    Double_t nevents=1.0, // overriden by nevhistoname
                    Double_t sigma=1.0, // sigma[pb]
                    Bool_t isParticlePlusAntiParticleYield=true,
                    Bool_t setUsePtDependentEffUncertainty=true,
                    const char *nevhistoname="hNEvents"){

  //
  // Get the histograms from the files
  //
  TH1D *hDirectMCpt=0;           // Input MC c-->D spectra
  TH1D *hFeedDownMCpt=0;         // Input MC b-->D spectra
  TH1D *hDirectMCptMax=0;        // Input MC maximum c-->D spectra
  TH1D *hDirectMCptMin=0;        // Input MC minimum c-->D spectra
  TH1D *hFeedDownMCptMax=0;      // Input MC maximum b-->D spectra
  TH1D *hFeedDownMCptMin=0;      // Input MC minimum b-->D spectra
  TGraphAsymmErrors * gFcConservative = 0; // Input fPrompt fraction
  AliHFSystErr * systematics = 0;

  TH1D *hDirectEffpt=0;          // c-->D Acceptance and efficiency correction
  TH1D *hFeedDownEffpt=0;        // b-->D Acceptance and efficiency correction
  TH1D *hRECpt=0;                // all reconstructed D

  //
  // Get theory predictions from cross section file for f_prompt
  //
  if(gSystem->Exec(Form("ls -l %s > /dev/null 2>&1",inputCrossSection)) !=0){
    printf("File %s with input fPrompt does not exist -> exiting\n",inputCrossSection);
    return;
  }
  TFile* inputcrossfile = new TFile(inputCrossSection,"read");
  if(!inputcrossfile){
    printf("File %s with input fPrompt not opened -> exiting\n",inputCrossSection);
    return;
  }

  hDirectMCpt = (TH1D*)inputcrossfile->Get("hDirectMCpt");
  hFeedDownMCpt = (TH1D*)inputcrossfile->Get("hFeedDownMCpt");
  hDirectMCptMax = (TH1D*)inputcrossfile->Get("hDirectMCptMax");
  hDirectMCptMin = (TH1D*)inputcrossfile->Get("hDirectMCptMin");
  hFeedDownMCptMax = (TH1D*)inputcrossfile->Get("hFeedDownMCptMax");
  hFeedDownMCptMin = (TH1D*)inputcrossfile->Get("hFeedDownMCptMin");
  gFcConservative = (TGraphAsymmErrors*)inputcrossfile->Get("gFcConservative");
  systematics = (AliHFSystErr*)inputcrossfile->Get("AliHFSystErr");

  //
  // Get efficiencies for cross section calculation
  //
  if(gSystem->Exec(Form("ls -l %s > /dev/null 2>&1",efffilename)) !=0){
    printf("File %s with efficiencies does not exist -> exiting\n",efffilename);
    return;
  }
  TFile * efffile = new TFile(efffilename,"read");
  if(!efffile){
    printf("File %s with efficiencies not opened -> exiting\n",efffilename);
    return;
  }
  hDirectEffpt = (TH1D*)efffile->Get(nameeffprompt);
  hDirectEffpt->SetNameTitle("hDirectEffpt","direct acc x eff");
  hFeedDownEffpt = (TH1D*)efffile->Get(nameefffeed);
  hFeedDownEffpt->SetNameTitle("hFeedDownEffpt","feed-down acc x eff");

  //
  // Get raw yield for cross section calculation
  //
  if(gSystem->Exec(Form("ls -l %s > /dev/null 2>&1",recofilename)) !=0){
    printf("File %s with raw yield does not exist -> exiting\n",recofilename);
    return;
  }
  TFile * recofile = new TFile(recofilename,"read");
  if(!recofile){
    printf("File %s with raw yields not opened -> exiting\n",recofilename);
    return;
  }
  hRECpt = (TH1D*)recofile->Get(recohistoname);
  hRECpt->SetNameTitle("hRECpt","Reconstructed spectra");

  //
  // Get (corrected) number of analysd events for cross section calculation
  //
  TH1F* hNorm=(TH1F*)recofile->Get(nevhistoname);
  if(hNorm){
    nevents=hNorm->GetBinContent(1);
  }else{
    printf("Histogram with number of events for norm not found in raw yiled file\n");
    printf("  nevents = %.0f will be used\n",nevents);
  }

  Int_t fnPtBins = hRECpt->GetNbinsX();
  Double_t *fPtBinLimits = new Double_t[fnPtBins+1];
  Double_t *fPtBinWidths = new Double_t[fnPtBins];
  Double_t xlow=0., binwidth=0.;
  for(Int_t i=1; i<fnPtBins; i++){
    binwidth = hRECpt->GetBinWidth(i);
    xlow = hRECpt->GetBinLowEdge(i);
    fPtBinLimits[i-1] = xlow;
    fPtBinWidths[i-1] = binwidth;
  }
  fPtBinLimits[fnPtBins] = xlow + binwidth;

  //
  // Define the remaining output histograms to be calculated here
  //
  TH1D *histoYieldCorr = new TH1D("histoYieldCorr","corrected yield",fnPtBins,fPtBinLimits);
  TH1D *histoYieldCorrMax = new TH1D("histoYieldCorrMax","max corrected yield (no feed-down corr)",fnPtBins,fPtBinLimits);
  TH1D *histoYieldCorrMin = new TH1D("histoYieldCorrMin","min corrected yield",fnPtBins,fPtBinLimits);
  TGraphAsymmErrors * gYieldCorr = new TGraphAsymmErrors(fnPtBins+1);
  TGraphAsymmErrors * gYieldCorrExtreme = new TGraphAsymmErrors(fnPtBins+1);
  TGraphAsymmErrors * gYieldCorrConservative = new TGraphAsymmErrors(fnPtBins+1);
  gYieldCorr->SetNameTitle("gYieldCorr","gYieldCorr (by Nb)");
  gYieldCorrExtreme->SetNameTitle("gYieldCorrExtreme","Extreme gYieldCorr (by Nb)");
  gYieldCorrConservative->SetNameTitle("gYieldCorrConservative","Conservative gYieldCorr (by Nb)");

  TH1D *histoSigmaCorr = new TH1D("histoSigmaCorr","corrected invariant cross-section",fnPtBins,fPtBinLimits);
  TH1D *histoSigmaCorrMax = new TH1D("histoSigmaCorrMax","max corrected invariant cross-section",fnPtBins,fPtBinLimits);
  TH1D *histoSigmaCorrMin = new TH1D("histoSigmaCorrMin","min corrected invariant cross-section",fnPtBins,fPtBinLimits);
  TGraphAsymmErrors * gSigmaCorr = new TGraphAsymmErrors(fnPtBins+1);
  TGraphAsymmErrors * gSigmaCorrExtreme = new TGraphAsymmErrors(fnPtBins+1);
  TGraphAsymmErrors * gSigmaCorrConservative = new TGraphAsymmErrors(fnPtBins+1);
  gSigmaCorr->SetNameTitle("gSigmaCorr","gSigmaCorr (by Nb)");
  gSigmaCorrExtreme->SetNameTitle("gSigmaCorrExtreme","Extreme gSigmaCorr (by Nb)");
  gSigmaCorrConservative->SetNameTitle("gSigmaCorrConservative","Conservative gSigmaCorr (by Nb)");

  // NB: Don't care for the moment about fhStatUncEffc/bSigma fhStatUncEffc/bFD
  TH1D *hStatUncEffcSigma = new TH1D("fhStatUncEffcSigma","direct charm stat unc on the cross section",fnPtBins,fPtBinLimits);
  TH1D *hStatUncEffbSigma = new TH1D("fhStatUncEffbSigma","secondary charm stat unc on the cross section",fnPtBins,fPtBinLimits);
  TH1D *hStatUncEffcFD = new TH1D("fhStatUncEffcFD","direct charm stat unc on the feed-down correction",fnPtBins,fPtBinLimits);
  TH1D *hStatUncEffbFD = new TH1D("fhStatUncEffbFD","secondary charm stat unc on the feed-down correction",fnPtBins,fPtBinLimits);

  //
  // Do the corrected yield calculation.
  // NB: Don't care for the moment about histoYieldCorrMin/Max gYieldCorrExtreme/Conservative
  // 
  Double_t value = 0., errvalue = 0., errvalueMax = 0., errvalueMin = 0.;
  for (Int_t ibin=1; ibin<=fnPtBins; ibin++) {
    // Calculate the value
    //    physics =  [ reco  - (lumi * delta_y * BR_b * eff_trig * eff_b * Nb_th) ] / bin-width
    //            =    reco * fprompt_NB
    Double_t frac = 1.0, errfrac =0.;

    // Variables initialization
    value = 0.; errvalue = 0.; errvalueMax = 0.; errvalueMin = 0.;

    // Get fPrompt from input value
    Double_t x = 0., correction = 0;
    gFcConservative->GetPoint(ibin,x,correction);

    // Calculate corrected yield (= raw yield * fprompt)
    if( hRECpt->GetBinContent(ibin)>0. && hRECpt->GetBinContent(ibin)!=0. && hFeedDownMCpt->GetBinContent(ibin)>0. && hFeedDownEffpt->GetBinContent(ibin)>0. ) value = hRECpt->GetBinContent(ibin) * correction;
    value /= hRECpt->GetBinWidth(ibin);
    if (value<0.) value =0.;

    //  Statistical uncertainty:   delta_physics = sqrt ( (delta_reco)^2 )  / bin-width
    if (value!=0. && hRECpt->GetBinError(ibin) && hRECpt->GetBinError(ibin)!=0.) errvalue = hRECpt->GetBinError(ibin);
    errvalue /= hRECpt->GetBinWidth(ibin);

    histoYieldCorr->SetBinContent(ibin,value);
    histoYieldCorr->SetBinError(ibin,errvalue);
    gYieldCorr->SetPoint(ibin,x,value);
    gYieldCorr->SetPointError(ibin,(fPtBinWidths[ibin-1]/2.),(fPtBinWidths[ibin-1]/2.),errvalueMin,errvalueMax);

  }

  //
  // Do the corrected sigma calculation.
  // NB: Don't care for the moment about histoSigmaCorrMin/Max and gSigmaCorrExtreme/Conservative
  // 
  Double_t fLuminosity[2] = {nevents / sigma, 0.04 * nevents / sigma};
  Double_t fTrigEfficiency[2] = {1.0, 0};
  Double_t fGlobalEfficiencyUncertainties[2] = {0.05, 0.05};
  Double_t deltaY = 1.0;
  Double_t branchingRatioC = 1.0;
  Double_t branchingRatioBintoFinalDecay = 1.0;
  Bool_t fGlobalEfficiencyPtDependent = setUsePtDependentEffUncertainty;
  Int_t fParticleAntiParticle = 1;
  if( isParticlePlusAntiParticleYield ) fParticleAntiParticle = 2;
  printf("\n\n     Correcting the spectra with : \n   luminosity = %2.2e +- %2.2e, trigger efficiency = %2.2e +- %2.2e, \n    delta_y = %2.2f, BR_c = %2.2e, BR_b_decay = %2.2e \n    %2.2f percent uncertainty on the efficiencies, and %2.2f percent uncertainty on the b/c efficiencies ratio \n    usage of pt-dependent efficiency uncertainty for Nb uncertainy calculation? %1.0d \n\n",fLuminosity[0],fLuminosity[1],fTrigEfficiency[0],fTrigEfficiency[1],deltaY,branchingRatioC,branchingRatioBintoFinalDecay,fGlobalEfficiencyUncertainties[0],fGlobalEfficiencyUncertainties[1],fGlobalEfficiencyPtDependent);

  // protect against null denominator
  if (deltaY==0. || fLuminosity[0]==0. || fTrigEfficiency[0]==0. || branchingRatioC==0.) {
    printf(" Hey you ! Why luminosity or trigger-efficiency or the c-BR or delta_y are set to zero ?! ");
    return;
  }

  for (Int_t ibin=1; ibin<=fnPtBins; ibin++) {

    // Variables initialization
    value=0.; errvalue=0.;

    Double_t x = histoYieldCorr->GetBinCenter(ibin);

    // Sigma calculation
    //   Sigma = ( 1. / (lumi * delta_y * BR_c * ParticleAntiPartFactor * eff_trig * eff_c ) ) * spectra (corrected for feed-down)
    if (hDirectEffpt->GetBinContent(ibin) && hDirectEffpt->GetBinContent(ibin)!=0. && hRECpt->GetBinContent(ibin)>0.) { 
      value = histoYieldCorr->GetBinContent(ibin) / ( deltaY * branchingRatioC * fParticleAntiParticle * fLuminosity[0] * fTrigEfficiency[0] * hDirectEffpt->GetBinContent(ibin) );
    }

    // Sigma statistical uncertainty:
    //   delta_sigma = sigma * sqrt ( (delta_spectra/spectra)^2 )
    if (value!=0.) {
      errvalue = value * (histoYieldCorr->GetBinError(ibin)/histoYieldCorr->GetBinContent(ibin));
    }

    histoSigmaCorr->SetBinContent(ibin,value);
    histoSigmaCorr->SetBinError(ibin,errvalue);
    gSigmaCorr->SetPoint(ibin,x,value); // i,x,y
    gSigmaCorr->SetPointError(ibin,(fPtBinWidths[ibin-1]/2.),(fPtBinWidths[ibin-1]/2.),errvalueMin,errvalueMax); // i,xl,xh,yl,yh
  }
  
  //
  // Define output file, in same style as original HFPtSpectrum macro
  //
  TFile *out = new TFile(outfilename,"recreate");
  out->cd();

  hDirectMCpt->Write();
  hFeedDownMCpt->Write();
  hDirectMCptMax->Write();
  hDirectMCptMin->Write();
  hFeedDownMCptMax->Write();
  hFeedDownMCptMin->Write();

  hDirectEffpt->Write();
  hFeedDownEffpt->Write();
  hRECpt->Write();

  histoYieldCorr->Write();
  histoYieldCorrMax->Write();
  histoYieldCorrMin->Write();

  histoSigmaCorr->Write();
  histoSigmaCorrMax->Write();
  histoSigmaCorrMin->Write();

  gYieldCorr->Write();
  gSigmaCorr->Write();
  gYieldCorrExtreme->Write();
  gSigmaCorrExtreme->Write();
  gYieldCorrConservative->Write();
  gSigmaCorrConservative->Write();

  gFcConservative->Write();

  hStatUncEffcSigma->Write();
  hStatUncEffbSigma->Write();
  hStatUncEffcFD->Write();
  hStatUncEffbFD->Write();

  systematics->Write();

  out->Close();
  recofile->Close();
  efffile->Close();
  inputcrossfile->Close();
}