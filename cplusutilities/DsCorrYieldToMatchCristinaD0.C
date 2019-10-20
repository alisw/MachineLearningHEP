


void DsCorrYieldToMatchCristinaD0(Int_t multBin=1, TString hfptspectrum="finalcrossDsppMBvspt_ntrklmult1.root", TString raw="masshisto.root"){
  
  
  // Ds case
  Float_t BR = 2.27e-2; 
  
  const int nPtBins = 6;
  
  TFile *fSpectrum = new TFile(hfptspectrum, "read"); // to get raw yield, efficiency and fprompt
  TFile *fNorm = new TFile(raw, "read");              // to get normalization (nbr of event)
  
  TH1D *hRawYield = (TH1D*)fSpectrum->Get("hRECpt");
  TH1D *hEff=(TH1D*)fSpectrum->Get("hDirectEffpt");
  TGraphAsymmErrors *gFprompt = (TGraphAsymmErrors*)fSpectrum->Get("gFcConservative");

  TH1D *hNorm = (TH1D*)fNorm->Get(Form("hEvForNorm_mult%d", multBin));
  Double_t Nev = hNorm->GetBinContent(2);


  // Compute corrected yield
  TH1D* corrY = (TH1D*)hRawYield->Clone("corrYield");

  for(Int_t ipt=0; ipt<nPtBins; ipt++){

    // raw yield
    double rawYield = hRawYield->GetBinContent(ipt+1);
    double rawYieldErr = hRawYield->GetBinError(ipt+1)/rawYield;
    // delta pt            
    double deltaPt =  hRawYield->GetBinWidth(ipt+1);

    // efficiency
    double eff = hEff->GetBinContent(ipt+1);

    // f prompt
    double x = 0.;
    double y_fprompt = 0.;
    gFprompt->GetPoint(ipt+1, x, y_fprompt);

    // corrected yield
    double corrYield = (1./deltaPt)*(1./BR)*(1./2.)*y_fprompt*(rawYield/eff)*(1./Nev);

    corrY->SetBinContent(ipt+1, corrYield);
    corrY->SetBinError(ipt+1, corrYield*sqrt(rawYieldErr*rawYieldErr));
  }
  
  // Save result
  TCanvas *cor = new TCanvas("cor","");
  corrY->SetTitle("corrYield");
  corrY->Draw();
  
  corrY->SaveAs(Form("Ds_corryield_mult%d.root", multBin));




}











