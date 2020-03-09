#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TFile.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2.h"
#include "TH2F.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TMath.h"
#include "TROOT.h"
#include "TStyle.h"
#include "AliHFSystErr.h"
#include <Riostream.h>
#endif

/* $Id$ */ 


//_________________________________________________________________________________________
//
//  Macro to combine the the MonteCarlo B feed-down subtraction uncertainties
//
//   Take as input the output files from the HFPtSpectrum class 
//    from both fc & NbScaled subtraction methods and combine the uncertainties. 
//   The final central value is set as the one from the Nb-method. 
//   The final uncertainties are defined as the envelope of both Nb & NbScaled
//      uncertainties with respect to the new central-value.
//   The final global uncertainties are also defined and a preliminary drawing done. 
//
//
//   Usage parameters:
//      1. HFPtSpectrum Nb subtraction file 
//      2. HFPtSpectrum NbScaled subtraction file 
//      3. Output file name
//      4. FONLL theoretical predictions file to draw on top
//      5. Decay channel as defined in the AliHFSystErr class
//
//_________________________________________________________________________________________



void CombineFeedDownMCSubtractionMethodsUncertainties(
					       const char *Nbfilename="HFPtSpectrum_LcpK0s_std_pp_13TeV_Nbx2_RunII_OPT.root",
					       const char *nbScaledfilename="HFPtSpectrum_LcpK0s_std_pp_13TeV_Nb_RunII_OPT.root",
				 	       const char *outfilename="HFPtSpectrum_LcpK0s_std_pp_13TeV_NbNbx2_RunII_OPT.root",
					       const char *thfilename="D0DplusDstarPredictions_13TeV_y05_all_300416_BDShapeCorrected.root",
					       Int_t decay=6)
{
  
 
  // Get Nb file inputs
  TFile * Nbfile = new TFile(Nbfilename,"read");
  TH1D * histoSigmaCorrNb = (TH1D*)Nbfile->Get("histoSigmaCorr");             
  histoSigmaCorrNb->SetNameTitle("histoSigmaCorrNb","histoSigmaCorrNb");
  TGraphAsymmErrors * gSigmaCorrNb = (TGraphAsymmErrors*)Nbfile->Get("gSigmaCorr");    
  gSigmaCorrNb->SetNameTitle("gSigmaCorrNb","gSigmaCorrNb");
  TGraphAsymmErrors * gSigmaCorrConservativeNb = (TGraphAsymmErrors*)Nbfile->Get("gSigmaCorrConservative"); 
  gSigmaCorrConservativeNb->SetNameTitle("gSigmaCorrConservativeNb","Cross section (Nb prompt fraction)");
  TGraphAsymmErrors * gNbConservativeNb = (TGraphAsymmErrors*)Nbfile->Get("gFcConservative");             
  gNbConservativeNb->SetNameTitle("gNbConservativeNb","Nb prompt fraction");

  // 
  // Get Nb Scaled file inputs
  TFile * nbScaledfile = new TFile(nbScaledfilename,"read");
  TH1D * histoSigmaCorrNbScaled = (TH1D*)nbScaledfile->Get("histoSigmaCorr");                            
  histoSigmaCorrNbScaled->SetNameTitle("histoSigmaCorrNbScaled","histoSigmaCorrNbScaled");
  TGraphAsymmErrors * gSigmaCorrNbScaled = (TGraphAsymmErrors*)nbScaledfile->Get("gSigmaCorr");
  gSigmaCorrNbScaled->SetNameTitle("gSigmaCorrNbScaled","gSigmaCorrNbScaled");
  TGraphAsymmErrors * gSigmaCorrConservativeNbScaled = (TGraphAsymmErrors*)nbScaledfile->Get("gSigmaCorrConservative");
  gSigmaCorrConservativeNbScaled->SetNameTitle("gSigmaCorrConservativeNbScaled","Cross section (Nb scaled (x2) prompt fraction)");
  TGraphAsymmErrors * gNbConservativeNbScaled = (TGraphAsymmErrors*)nbScaledfile->Get("gFcConservative");
  gNbConservativeNbScaled->SetNameTitle("gNbConservativeNbScaled","Nb scaled (x2) prompt fraction");

  //
  // Get the predictions input
  TFile *thfile = new TFile(thfilename,"read");
  TGraphAsymmErrors * thD0KpifromBprediction = (TGraphAsymmErrors*)thfile->Get("D0Kpiprediction");
  TGraphAsymmErrors * thDpluskpipiprediction = (TGraphAsymmErrors*)thfile->Get("Dpluskpipiprediction");
  TGraphAsymmErrors * thDstarD0piprediction = (TGraphAsymmErrors*)thfile->Get("DstarD0piprediction");
  TGraphAsymmErrors * thDsKKpiprediction = (TGraphAsymmErrors*)thfile->Get("DsKkpiprediction");
  TGraphAsymmErrors * thLcpK0Sprediction = (TGraphAsymmErrors*)thfile->Get("LcK0spprediction");

  thD0KpifromBprediction->SetLineColor(4);
  thD0KpifromBprediction->SetFillColor(kAzure+9);
  thDpluskpipiprediction->SetLineColor(4);
  thDpluskpipiprediction->SetFillColor(kAzure+9);
  thDstarD0piprediction->SetLineColor(4);
  thDstarD0piprediction->SetFillColor(kAzure+9);
  thDsKKpiprediction->SetLineColor(4);
  thDsKKpiprediction->SetFillColor(kAzure+9);
  thLcpK0Sprediction->SetLineColor(6);
  thLcpK0Sprediction->SetFillColor(kAzure+6);

  //
  // Get the spectra bins & limits
  Int_t nbins = histoSigmaCorrNb->GetNbinsX();
  Double_t *limits = new Double_t[nbins+1];
  Double_t xlow=0., binwidth=0.;
  for (Int_t i=1; i<=nbins; i++) {
    binwidth = histoSigmaCorrNb->GetBinWidth(i);
    xlow = histoSigmaCorrNb->GetBinLowEdge(i);
    limits[i-1] = xlow;
  }
  limits[nbins] = xlow + binwidth;


  //
  // Define a new histogram with the real-data reconstructed spectrum binning 
  //   they will be filled with central value equal to the Nb result
  //   and uncertainties taken from the envelope of the result uncertainties
  //   The systematical unc. (but FD) will also be re-calculated
  //
  TH1D * histoSigmaCorr = new TH1D("histoSigmaCorr","corrected cross-section (combined Nb and NbScaled MC feed-down subtraction)",nbins,limits);
  histoSigmaCorr->GetXaxis()->SetTitle("p_{T}  [GeV]");
  TGraphAsymmErrors * gSigmaCorr = new TGraphAsymmErrors(nbins+1);
  gSigmaCorr->SetNameTitle("gSigmaCorr","gSigmaCorr (combined Nb and NbScaled MC FD)");
  TGraphAsymmErrors * gNbCorrConservative = new TGraphAsymmErrors(nbins+1);
  gNbCorrConservative->SetNameTitle("gNbCorrConservative","Combined (Nb,Nbx2) prompt fraction");
  TGraphAsymmErrors * gSigmaCorrConservative = new TGraphAsymmErrors(nbins+1);
  gSigmaCorrConservative->SetNameTitle("gSigmaCorrConservative","Cross section (combined (Nb,Nbx2) prompt fraction)");
  TGraphAsymmErrors * gSigmaCorrConservativePC = new TGraphAsymmErrors(nbins+1);
  gSigmaCorrConservativePC->SetNameTitle("gSigmaCorrConservativePC","Conservative gSigmaCorr (combined Nb and NbScaled MC FD) in percentages [for drawing with AliHFSystErr]");



  // 
  // Loop on all the bins to do the calculations
  //
  Double_t pt=0., average = 0., averageStatUnc=0., avErrx=0., avErryl=0., avErryh=0., avErryfdl=0., avErryfdh=0.;
  Double_t avErrylPC=0., avErryhPC=0., avErryfdlPC=0., avErryfdhPC=0.;
  Double_t valNb = 0., valNbErrstat=0., valNbErrx=0., valNbErryl=0., valNbErryh=0., valNbErryfdl=0., valNbErryfdh=0.;
  Double_t valNbScaled = 0., valNbScaledErrstat=0., valNbScaledErrx=0., valNbScaledErryl=0., valNbScaledErryh=0., valNbScaledErryfdl=0., valNbScaledErryfdh=0.;
  Double_t corrfd = 0., corrfdl=0., corrfdh=0.;
  //
  for(Int_t ibin=1; ibin<=nbins; ibin++){

    // Get input values from Nb method
    valNb = histoSigmaCorrNb->GetBinContent(ibin);
    pt = histoSigmaCorrNb->GetBinCenter(ibin);
    valNbErrstat = histoSigmaCorrNb->GetBinError(ibin);
    Double_t value =0., ptt=0.;
    gSigmaCorrConservativeNb->GetPoint(ibin,ptt,value);
    if (value<=0.) continue;
    if ( TMath::Abs(valNb-value)>0.1 || TMath::Abs(pt-ptt)>0.1 ) 
      cout << "Hey you ! There might be a problem with the Nb input file, please, have a look !" << endl;
    valNbErrx = gSigmaCorrNb->GetErrorXlow(ibin);
    valNbErryl = gSigmaCorrNb->GetErrorYlow(ibin);
    valNbErryh = gSigmaCorrNb->GetErrorYhigh(ibin);
    valNbErryfdl = TMath::Abs( gSigmaCorrConservativeNb->GetErrorYlow(ibin) );
    valNbErryfdh = TMath::Abs( gSigmaCorrConservativeNb->GetErrorYhigh(ibin) );
    Double_t valfdNb = 0., x=0.;
    gNbConservativeNb->GetPoint(ibin,x,valfdNb);
    Double_t valfdNbh = gNbConservativeNb->GetErrorYhigh(ibin);
    Double_t valfdNbl = gNbConservativeNb->GetErrorYlow(ibin);

    // Get input values from NbScaled method
    valNbScaled = histoSigmaCorrNbScaled->GetBinContent(ibin);
    pt = histoSigmaCorrNbScaled->GetBinCenter(ibin);
    valNbScaledErrstat = histoSigmaCorrNbScaled->GetBinError(ibin);
    gSigmaCorrConservativeNbScaled->GetPoint(ibin,ptt,value);
    if ( TMath::Abs(valNbScaled-value)>0.1 || TMath::Abs(pt-ptt)>0.1 ) 
      cout << "Hey you ! There might be a problem with the NbScaled input file, please, have a look !" << endl;
    valNbScaledErrx = gSigmaCorrNbScaled->GetErrorXlow(ibin);
    valNbScaledErryl = gSigmaCorrNbScaled->GetErrorYlow(ibin);
    valNbScaledErryh = gSigmaCorrNbScaled->GetErrorYhigh(ibin);
    valNbScaledErryfdl = gSigmaCorrConservativeNbScaled->GetErrorYlow(ibin);
    valNbScaledErryfdh = gSigmaCorrConservativeNbScaled->GetErrorYhigh(ibin);
    Double_t valfdNbScaled = 0.;
    gNbConservativeNbScaled->GetPoint(ibin,x,valfdNbScaled);
    Double_t valfdNbScaledh = gNbConservativeNbScaled->GetErrorYhigh(ibin);
    Double_t valfdNbScaledl = gNbConservativeNbScaled->GetErrorYlow(ibin);
    

    // Compute the FD combined value
    //    average = valNbScaled
    average = valNbScaled ;
    corrfd = valfdNbScaled;
    avErrx = valNbErrx;
    if ( TMath::Abs( valNbErrx - valNbScaledErrx ) > 0.1 ) 
      cout << "Hey you ! There might be consistency problem with the Nb & NbScaled input files, please, have a look !" << endl;
    averageStatUnc = valNbScaledErrstat ;
//     cout << " pt=" << pt << ", average="<<average<<endl;
//     cout << "   stat unc (pc)=" << averageStatUnc/average << ", stat-Nb (pc)="<<(valNbErrstat/valNb) << ", stat-NbScaled (pc)="<<(valNbScaledErrstat/valNbScaled)<<endl;
    
    // now estimate the new feed-down combined uncertainties
    Double_t minimum[2] = { (valNb - valNbErryfdl), (valNbScaled - valNbScaledErryfdl) };
    Double_t maximum[2] = { (valNb + valNbErryfdh), (valNbScaled + valNbScaledErryfdh) };
    avErryfdl = average - TMath::MinElement(2,minimum);
    avErryfdh = TMath::MaxElement(2,maximum) - average;
    avErryfdlPC = avErryfdl / average ; // in percentage
    avErryfdhPC = avErryfdh / average ; // in percentage
//     cout << " Nb : val " << valNb << " + " << valNbErryfdh <<" - " << valNbErryfdl <<endl;
//     cout << " NbScaled : val " << valNbScaled << " + " << valNbScaledErryfdh <<" - " << valNbScaledErryfdl <<endl;
//     cout << " Nb  & NbScaled: val " << average << " + " << avErryfdh <<" - " << avErryfdl <<endl;
    Double_t minimumNb[2] = { (valfdNbScaled - valfdNbScaledl), (valfdNb - valfdNbl) };
    Double_t maximumNb[2] = { (valfdNbScaled + valfdNbScaledh), (valfdNb + valfdNbh) };
    corrfdl = corrfd - TMath::MinElement(2,minimumNb);
    corrfdh = TMath::MaxElement(2,maximumNb) - corrfd;


   

    // fill in the histos and TGraphs
    //   fill them only when for non empty bins
    if ( average > 0.1 ) {
      histoSigmaCorr->SetBinContent(ibin,average);
      histoSigmaCorr->SetBinError(ibin,averageStatUnc);
      gSigmaCorr->SetPoint(ibin,pt,average);
      gSigmaCorr->SetPointError(ibin,valNbErrx,valNbErrx,avErryl,avErryh);
      gSigmaCorrConservative->SetPoint(ibin,pt,average);
      gSigmaCorrConservative->SetPointError(ibin,valNbErrx,valNbErrx,avErryfdl,avErryfdh);
      gSigmaCorrConservativePC->SetPoint(ibin,pt,0.);
      gSigmaCorrConservativePC->SetPointError(ibin,valNbErrx,valNbErrx,avErryfdlPC,avErryfdhPC);
      gNbCorrConservative->SetPoint(ibin,pt,corrfd);
      gNbCorrConservative->SetPointError(ibin,valNbErrx,valNbErrx,corrfdl,corrfdh);
    }

  }


  gROOT->SetStyle("Plain");
  gStyle->SetOptTitle(0);

  //
  // Plot the results
  TH2F *histo2Draw = new TH2F("histo2Draw","histo2 (for drawing)",100,2,12.,100,1e3,5e7);
  histo2Draw->SetStats(0);
  histo2Draw->GetXaxis()->SetTitle("p_{T}  [GeV]");
  histo2Draw->GetXaxis()->SetTitleSize(0.05);
  histo2Draw->GetXaxis()->SetTitleOffset(0.95);
  histo2Draw->GetYaxis()->SetTitle("#frac{1}{BR} #times #frac{d#sigma}{dp_{T}} |_{|y|<0.5}");
  histo2Draw->GetYaxis()->SetTitleOffset(1.29);
  histo2Draw->GetYaxis()->SetTitleSize(0.035);
  //
  TCanvas *combinefdunc = new TCanvas("combinefdunc","show the FD results combination");
  //
  histo2Draw->Draw();
  //
  histoSigmaCorrNb->SetMarkerStyle(20);
  histoSigmaCorrNb->SetMarkerColor(kGreen+2);
  histoSigmaCorrNb->SetLineColor(kGreen+2);
  histoSigmaCorrNb->Draw("esame");
  gSigmaCorrConservativeNb->SetMarkerStyle(20);
  gSigmaCorrConservativeNb->SetMarkerColor(kGreen+2);
  gSigmaCorrConservativeNb->SetLineColor(kGreen+2);
  gSigmaCorrConservativeNb->SetFillStyle(3004);//2);
  gSigmaCorrConservativeNb->SetFillColor(kGreen);
  //////////gSigmaCorrConservativeNb->Draw("2[]same");
  //
  histoSigmaCorrNbScaled->SetMarkerStyle(25);
  histoSigmaCorrNbScaled->SetMarkerColor(kViolet+5);
  histoSigmaCorrNbScaled->SetLineColor(kViolet+5);
  histoSigmaCorrNbScaled->Draw("esame");
  gSigmaCorrConservativeNbScaled->SetMarkerStyle(25);
  gSigmaCorrConservativeNbScaled->SetMarkerColor(kOrange+7);//kViolet+5);
  gSigmaCorrConservativeNbScaled->SetLineColor(kOrange+7);//kOrange+7);//kViolet+5);
  gSigmaCorrConservativeNbScaled->SetFillStyle(3018);//02);
  gSigmaCorrConservativeNbScaled->SetFillColor(kMagenta);
  //////////gSigmaCorrConservativeNbScaled->Draw("2[]same");
  //
  gSigmaCorrConservative->SetLineColor(kRed);
  gSigmaCorrConservative->SetLineWidth(2);
  gSigmaCorrConservative->SetFillColor(kRed);
  gSigmaCorrConservative->SetFillStyle(0);
  gSigmaCorrConservative->Draw("2");
  histoSigmaCorr->SetMarkerColor(kRed);
  //////////histoSigmaCorr->Draw("esame");
  //
  //
  TLegend* leg=combinefdunc->BuildLegend();
  leg->SetFillStyle(0);
  combinefdunc->SetLogy();
  combinefdunc->Update();

  TCanvas *combineNbunc = new TCanvas("combineNbunc","show the Nb FD results combination");
  //
  TH2F *histo3Draw = new TH2F("histo3Draw","histo3 (for drawing)",100,0,24.,10,0.,1.);
  histo3Draw->SetStats(0);
  histo3Draw->GetXaxis()->SetTitle("p_{T}  [GeV]");
  histo3Draw->GetXaxis()->SetTitleSize(0.05);
  histo3Draw->GetXaxis()->SetTitleOffset(0.95);
  histo3Draw->GetXaxis()->SetRangeUser(1.,24.);
  histo3Draw->GetYaxis()->SetTitle("Prompt fraction of the raw yields");
  histo3Draw->GetYaxis()->SetTitleSize(0.05);
  histo3Draw->GetYaxis()->SetRangeUser(0.,1.1);
  histo3Draw->Draw();
  //
  gNbConservativeNb->SetMarkerStyle(20);
  gNbConservativeNb->SetMarkerColor(kGreen+2);
  gNbConservativeNb->SetLineColor(kGreen+2);
  gNbConservativeNb->SetFillStyle(3004);
  gNbConservativeNb->SetFillColor(kGreen);
  gNbConservativeNb->Draw("2P");
  //
  gNbConservativeNbScaled ->SetMarkerStyle(22);
  gNbConservativeNbScaled ->SetMarkerSize(1.3);
  gNbConservativeNbScaled->SetMarkerColor(kOrange+7);//kViolet+5);
  gNbConservativeNbScaled->SetLineColor(kOrange+7);//kViolet+5);
  gNbConservativeNbScaled->SetFillStyle(3018);
  gNbConservativeNbScaled->SetFillColor(kOrange);
  gNbConservativeNbScaled->Draw("2P");
  //
  gNbCorrConservative->SetMarkerStyle(21);
  //gNbCorrConservative->SetMarkerColor(kRed);
  gNbCorrConservative->SetLineColor(kRed);
  gNbCorrConservative->SetLineWidth(2);
  gNbCorrConservative->SetFillColor(kRed);
  gNbCorrConservative->SetFillStyle(0);
  gNbCorrConservative->Draw("2P");
  //
  leg=combineNbunc->BuildLegend(0.65,0.75,0.9,0.9);
  leg->SetFillStyle(0);
  //
  combineNbunc->Update();

  //
  // Plot the results
  TCanvas *finalresults = new TCanvas("finalresults","show all combined results");
  //
  if ( decay==1 ) {
    thD0KpifromBprediction->SetLineColor(kGreen+2);
    thD0KpifromBprediction->SetLineWidth(3);
    thD0KpifromBprediction->SetFillColor(kGreen-6);
    thD0KpifromBprediction->Draw("3CA");
    thD0KpifromBprediction->Draw("CX");
  }
  else if ( decay==2 ) {
    thDpluskpipiprediction->SetLineColor(kGreen+2);
    thDpluskpipiprediction->SetLineWidth(3);
    thDpluskpipiprediction->SetFillColor(kGreen-6);
    thDpluskpipiprediction->Draw("3CA");
    thDpluskpipiprediction->Draw("CX");
  }
  else if ( decay==3 ) {
    thDstarD0piprediction->SetLineColor(kGreen+2);
    thDstarD0piprediction->SetLineWidth(3);
    thDstarD0piprediction->SetFillColor(kGreen-6);
    thDstarD0piprediction->Draw("3CA");
    thDstarD0piprediction->Draw("CX");
  }
  else if ( decay==4 ) {
    thDsKKpiprediction->SetLineColor(kGreen+2);
    thDsKKpiprediction->SetLineWidth(3);
    thDsKKpiprediction->SetFillColor(kGreen-6);
    thDsKKpiprediction->Draw("3CA");
    thDsKKpiprediction->Draw("CX");
  }
  else if ( decay==6 ) {
    thLcpK0Sprediction->SetLineColor(kGreen+2);
    thLcpK0Sprediction->SetLineWidth(3);
    thLcpK0Sprediction->SetFillColor(kGreen-6);
    thLcpK0Sprediction->Draw("3CA");
    thLcpK0Sprediction->Draw("CX");
  }
  //
  gSigmaCorr->SetLineColor(kRed);
  gSigmaCorr->SetLineWidth(1);
  gSigmaCorr->SetFillColor(kRed);
  gSigmaCorr->SetFillStyle(0);
  gSigmaCorr->Draw("2");
  histoSigmaCorr->SetMarkerStyle(21);
  histoSigmaCorr->SetMarkerColor(kRed);
  histoSigmaCorr->Draw("esame");
  //
  leg = new TLegend(0.7,0.75,0.87,0.5);
  leg->SetBorderSize(0);
  leg->SetLineColor(0);
  leg->SetFillColor(0);
  leg->SetTextFont(42);
  if ( decay==1 ) leg->AddEntry(thD0KpifromBprediction,"FONLL ","fl");
  else if ( decay==2 ) leg->AddEntry(thDpluskpipiprediction,"FONLL ","fl");
  else if ( decay==3 ) leg->AddEntry(thDstarD0piprediction,"FONLL ","fl");
  else if ( decay==4 ) leg->AddEntry(thDsKKpiprediction,"FONLL ","fl");
  else if ( decay==6 ) leg->AddEntry(thLcpK0Sprediction,"FONLL ","fl");
  leg->AddEntry(histoSigmaCorr,"data stat. unc.","pl");
  leg->AddEntry(gSigmaCorr,"data syst. unc.","f");
  leg->Draw();
  //
  finalresults->SetLogy();
  finalresults->Update();


 

  // Write the output to a file
 
  TFile * out = new TFile(outfilename,"recreate");
  histoSigmaCorr->Write();
  gSigmaCorr->Write();
  gSigmaCorrConservative->Write();
  gSigmaCorrConservativePC->Write();
  gNbCorrConservative->Write();
  out->Write();

}
