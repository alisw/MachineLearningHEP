//###################################################################################//
// Brief: Macro for invariant-mass spectra fits                                      //
// Main Function: mass_fitter                                                        //
// Author: Fabrizio Grosa, fabrizio.grosa@cern.ch                                    //
// Before running this macro make sure you have installed yaml-cpp on your laptop:   //
// MAC OSX --> brew install yaml-cpp                                                 //
// Ubuntu --> apt install yaml-cpp                                                   //
//###################################################################################//

#if !defined (__CINT__) || defined (__CLING__)

#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "Riostream.h"
#include "TH1.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TMath.h"

#include "AliHFInvMassFitter.h"
#include "AliVertexingHFUtils.h"

#endif

using namespace std;

//__________________________________________________________________________________________________________________
void mass_fitter(TString mcordata, TString useml);
double SingleGaus(double *m, double *pars);
double DoublePeakSingleGaus(double *x, double *pars);
double DoubleGaus(double *m, double *pars);
double DoublePeakDoubleGaus(double *m, double *pars);
void DivideCanvas(TCanvas* c, int nptbins);

//__________________________________________________________________________________________________________________
void mass_fitter(TString mcordata, TString useml) {

    //load config
    YAML::Node config = YAML::LoadFile("../machine_learning_hep/submission/default_complete.yml");
    bool uselikelihood = config["analysis"]["uselikelihood"].as<int>();
    vector<double> ptmin = config["analysis"]["binmin"].as<vector<double>>();
    vector<double> ptmax = config["analysis"]["binmax"].as<vector<double>>();
    double massforfit = config["analysis"]["masspeak"].as<double>();
    vector<double> massmin = config["analysis"]["massmin"].as<vector<double>>();
    vector<double> massmax = config["analysis"]["massmax"].as<vector<double>>();
    vector<int> rebin = config["analysis"]["rebin"].as<vector<int>>();
    vector<int> includesecpeak = config["analysis"]["includesecpeak"].as<vector<int>>();
    double masssecpeak = config["analysis"]["masssecpeak"].as<double>();
    vector<string> sbkgfunc = config["analysis"]["bkgfunc"].as<vector<string>>();
    vector<string> ssgnfunc = config["analysis"]["sgnfunc"].as<vector<string>>();
    const unsigned int nptbins = ptmin.size();
    int bkgfunc[nptbins], sgnfunc[nptbins];
    double ptlims[nptbins+1];

    for(unsigned int ipt=0; ipt<nptbins; ipt++) {
        ptlims[ipt] = ptmin[ipt];
        ptlims[ipt+1] = ptmax[ipt];

        if(sbkgfunc[ipt] == "kExpo")
            bkgfunc[ipt] = AliHFInvMassFitter::kExpo;
        else if(sbkgfunc[ipt] == "kLin")
            bkgfunc[ipt] = AliHFInvMassFitter::kLin;
        else if(sbkgfunc[ipt] == "kPol2")
            bkgfunc[ipt] = AliHFInvMassFitter::kPol2;
        else {
            cerr << "Bkg fit function not defined! Exit" << endl;
            return;
        }

        if(ssgnfunc[ipt] == "kGaus")
            sgnfunc[ipt] = AliHFInvMassFitter::kGaus;
        else if(ssgnfunc[ipt] == "k2Gaus")
            sgnfunc[ipt] = AliHFInvMassFitter::k2Gaus;
        else {
            cerr << "Signal fit function not defined! Exit" << endl;
            return;
        }
    }

    //load inv-mass histos
    auto infile = TFile::Open(Form("roottotal%s%s.root",mcordata.Data(), useml.Data()));
    if(!infile || !infile->IsOpen())
        return;
    TH1F* hMass[nptbins];
    for(unsigned int ipt=0; ipt<nptbins; ipt++) {
        hMass[ipt] = static_cast<TH1F*>(infile->Get(Form("h_invmass%0.f_%0.f",ptmin[ipt],ptmax[ipt])));
        hMass[ipt]->SetDirectory(0);
    }
    infile->Close();

    //define output histos
    auto hRawYields = new TH1D("hRawYields",";#it{p}_{T} (GeV/#it{c});raw yield",nptbins,ptlims);
    auto hSigma = new TH1D("hSigma",";#it{p}_{T} (GeV/#it{c});width (GeV/#it{c}^{2})",nptbins,ptlims);
    auto hMean = new TH1D("hMean",";#it{p}_{T} (GeV/#it{c});mean (GeV/#it{c}^{2})",nptbins,ptlims);
    auto hSoverB = new TH1D("hSoverB",";#it{p}_{T} (GeV/#it{c});S/B (3#sigma)",nptbins,ptlims);
    auto hSignal = new TH1D("hSignal",";#it{p}_{T} (GeV/#it{c});Signal (3#sigma)",nptbins,ptlims);
    auto hBkg = new TH1D("hBkg",";#it{p}_{T} (GeV/#it{c});Background (3#sigma)",nptbins,ptlims);
    auto hSignificance = new TH1D("hSignificance",";#it{p}_{T} (GeV/#it{c});Significance (3#sigma)",nptbins,ptlims);
    auto hChiSquare = new TH1D("hChiSquare",";#it{p}_{T} (GeV/#it{c});#chi^{2}/#it{ndf}",nptbins,ptlims);

    //fit histos
    TH1F* hMassForFit[nptbins];
    TCanvas* cMass = new TCanvas("cMass","cMass",1920,1080);
    DivideCanvas(cMass,nptbins);
    for(unsigned int ipt=0; ipt<nptbins; ipt++) {

        hMassForFit[ipt]=reinterpret_cast<TH1F*>(AliVertexingHFUtils::RebinHisto(hMass[ipt],rebin[ipt]));
        hMassForFit[ipt]->SetTitle(Form("%0.f < #it{p}_{T} < %0.f GeV/#it{c}; #it{M} (GeV/#it{c}^{2});Counts per %0.f MeV/#it{c}^{2}",ptmin[ipt],ptmax[ipt],hMassForFit[ipt]->GetBinWidth(1)*1000));
        hMassForFit[ipt]->SetName(Form("MassForFit%d",ipt));

        if(mcordata=="mc") { //MC
            int parrawyield = 0, parmean = 1., parsigma = 2;
            TF1* massfunc = NULL;
            if(sgnfunc[ipt]==AliHFInvMassFitter::kGaus) {
                if(!(includesecpeak[ipt])) {
                    massfunc = new TF1(Form("massfunc%d",ipt),SingleGaus,massmin[ipt],massmax[ipt],3);
                    massfunc->SetParameters(hMassForFit[ipt]->Integral()*hMassForFit[ipt]->GetBinWidth(1),massforfit,0.010);
                }
                else {
                    massfunc = new TF1(Form("massfunc%d",ipt),DoublePeakSingleGaus,massmin[ipt],massmax[ipt],6);
                    massfunc->SetParameters(hMassForFit[ipt]->Integral()*hMassForFit[ipt]->GetBinWidth(1),massforfit,0.010,hMassForFit[ipt]->Integral()*hMassForFit[ipt]->GetBinWidth(1),masssecpeak,0.010);
                }
            }
            else if(sgnfunc[ipt]==AliHFInvMassFitter::k2Gaus){
                if(!(includesecpeak[ipt])) {
                    massfunc = new TF1(Form("massfunc%d",ipt),DoubleGaus,massmin[ipt],massmax[ipt],5);
                    massfunc->SetParameters(hMassForFit[ipt]->Integral()*hMassForFit[ipt]->GetBinWidth(1),massforfit,0.010,0.030,0.9);
                }
                else {
                    massfunc = new TF1(Form("massfunc%d",ipt),DoublePeakDoubleGaus,massmin[ipt],massmax[ipt],8);
                    massfunc->SetParameters(hMassForFit[ipt]->Integral()*hMassForFit[ipt]->GetBinWidth(1),massforfit,0.010,0.030,0.9,hMassForFit[ipt]->Integral()*hMassForFit[ipt]->GetBinWidth(1),masssecpeak,0.010);
                }
            }
            else {
                cerr << "Fit function for MC not defined! Exit" << endl;
                return;
            }

            if(nptbins>1)
                cMass->cd(ipt+1);
            else
                cMass->cd();
            hMassForFit[ipt]->Fit(massfunc,"E"); //fit with chi2

            hRawYields->SetBinContent(ipt+1,massfunc->GetParameter(parrawyield));
            hRawYields->SetBinError(ipt+1,massfunc->GetParError(parrawyield));
            hSigma->SetBinContent(ipt+1,massfunc->GetParameter(parsigma));
            hSigma->SetBinError(ipt+1,massfunc->GetParError(parsigma));
            hMean->SetBinContent(ipt+1,massfunc->GetParameter(parmean));
            hMean->SetBinError(ipt+1,massfunc->GetParError(parmean));
            hChiSquare->SetBinContent(ipt+1,massfunc->GetChisquare() / massfunc->GetNDF());
            hChiSquare->SetBinError(ipt+1,0.);
        }
        else { //data
            auto massFitter = new AliHFInvMassFitter(hMassForFit[ipt],massmin[ipt],massmax[ipt],bkgfunc[ipt],sgnfunc[ipt]);
            if(uselikelihood) massFitter->SetUseLikelihoodFit();
            massFitter->SetInitialGaussianMean(massforfit);
            massFitter->SetInitialGaussianSigma(0.010);

            if(includesecpeak[ipt])
                massFitter->IncludeSecondGausPeak(masssecpeak,false,0.008,true);
            bool fitok = massFitter->MassFitter(false);

            double rawyield = massFitter->GetRawYield();
            double rawyielderr = massFitter->GetRawYieldError();
            double sigma = massFitter->GetSigma();
            double sigmaerr = massFitter->GetSigmaUncertainty();
            double mean = massFitter->GetMean();
            double meanerr = massFitter->GetMeanUncertainty();
            double redchi2 = massFitter->GetReducedChiSquare();
            double signif=0., signiferr=0.;
            double sgn=0., sgnerr=0.;
            double bkg=0., bkgerr=0.;
            massFitter->Significance(3,signif,signiferr);
            massFitter->Signal(3,sgn,sgnerr);
            massFitter->Background(3,bkg,bkgerr);

            hRawYields->SetBinContent(ipt+1,rawyield);
            hRawYields->SetBinError(ipt+1,rawyielderr);
            hSigma->SetBinContent(ipt+1,sigma);
            hSigma->SetBinError(ipt+1,sigmaerr);
            hMean->SetBinContent(ipt+1,mean);
            hMean->SetBinError(ipt+1,meanerr);
            hSignificance->SetBinContent(ipt+1,signif);
            hSignificance->SetBinError(ipt+1,signiferr);
            hSoverB->SetBinContent(ipt+1,sgn/bkg);
            hSoverB->SetBinError(ipt+1,sgn/bkg*TMath::Sqrt(sgnerr/sgn*sgnerr/sgn+bkgerr/bkg*bkgerr/bkg));
            hSignal->SetBinContent(ipt+1,sgn);
            hSignal->SetBinError(ipt+1,sgnerr);
            hBkg->SetBinContent(ipt+1,bkg);
            hBkg->SetBinError(ipt+1,bkgerr);
            hChiSquare->SetBinContent(ipt+1,redchi2);
            hChiSquare->SetBinError(ipt+1,1.e-20);

            if(nptbins>1)
                cMass->cd(ipt+1);
            else
                cMass->cd();

            hMassForFit[ipt]->GetYaxis()->SetRangeUser(hMassForFit[ipt]->GetMinimum()*0.95,hMassForFit[ipt]->GetMaximum()*1.2);
            if(!fitok)
                massFitter->GetHistoClone()->Draw();
            else
                massFitter->DrawHere(gPad);
        }
        cMass->Modified();
        cMass->Update();
    }

    //save output histos
    TString outfilename = Form("rawyields%s%s.root",mcordata.Data(), useml.Data());
    TFile outfile(outfilename.Data(),"recreate");
    cMass->Write();
    for(unsigned int ipt=0; ipt<nptbins; ipt++)
        hMass[ipt]->Write();
    hRawYields->Write();
    hSigma->Write();
    hMean->Write();
    hSignificance->Write();
    hSoverB->Write();
    hSignal->Write();
    hBkg->Write();
    hChiSquare->Write();
    outfile.Close();

    outfilename.ReplaceAll(".root",".pdf");
    cMass->SaveAs(outfilename.Data());
}

//__________________________________________________________________________________________________________________
double SingleGaus(double *m, double *pars) {
  double norm = pars[0], mean = pars[1], sigma = pars[2];

  return norm*TMath::Gaus(m[0],mean,sigma,true);
}

//__________________________________________________________________________________________________________________
double DoubleGaus(double *m, double *pars) {
  double norm = pars[0], mean = pars[1], sigma1 = pars[2], sigma1_2 = pars[3], fg = pars[4];

  return norm*((1-fg)*TMath::Gaus(m[0],mean,sigma1,true)+fg*TMath::Gaus(m[0],mean,sigma1_2,true));
}

//__________________________________________________________________________________________________________________
double DoublePeakSingleGaus(double *m, double *pars) {
  double norm1 = pars[0], mean1 = pars[1], sigma1 = pars[2]; //Ds peak
  double norm2 = pars[3], mean2 = pars[4], sigma2 = pars[5]; //Dplus peak for Ds

  return norm1*TMath::Gaus(m[0],mean1,sigma1,true) + norm2*TMath::Gaus(m[0],mean2,sigma2,true);
}

//__________________________________________________________________________________________________________________
double DoublePeakDoubleGaus(double *m, double *pars) {
  double norm1 = pars[0], mean = pars[1], sigma1 = pars[2], sigma1_2 = pars[3], fg = pars[4]; //Ds peak
  double norm2 = pars[5], mean2 = pars[6], sigma2 = pars[7]; //Dplus peak for Ds

  return norm1*((1-fg)*TMath::Gaus(m[0],mean,sigma1,true)+fg*TMath::Gaus(m[0],mean,sigma1_2,true)) + norm2*TMath::Gaus(m[0],mean2,sigma2,true);
}

//__________________________________________________________________________________________________________________
void DivideCanvas(TCanvas* c, int nptbins) {
  if(nptbins<2)
    c->cd();
  else if(nptbins==2 || nptbins==3)
    c->Divide(nptbins,1);
  else if(nptbins==4 || nptbins==6 || nptbins==8)
    c->Divide(nptbins/2,2);
  else if(nptbins==5 || nptbins==7)
    c->Divide((nptbins+1)/2,2);
  else if(nptbins==9 || nptbins==12 || nptbins==15)
    c->Divide(nptbins/3,3);
  else if(nptbins==10 || nptbins==11)
    c->Divide(4,3);
  else if(nptbins==13 || nptbins==14)
    c->Divide(5,3);
  else if(nptbins>15 && nptbins<=20 && nptbins%4==0)
    c->Divide(nptbins/4,4);
  else if(nptbins>15 && nptbins<=20 && nptbins%4!=0)
    c->Divide(5,4);
  else if(nptbins==21)
    c->Divide(7,3);
  else if(nptbins>21 && nptbins<=25)
    c->Divide(5,5);
  else if(nptbins>25 && nptbins%2==0)
    c->Divide(nptbins/2,2);
  else
    c->Divide((nptbins+1)/2,2);
}
