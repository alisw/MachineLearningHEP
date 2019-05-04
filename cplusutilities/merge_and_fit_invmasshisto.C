/*
 Macro to fit invariant mass distributions
 1) input files:
    masshistoLctopK0sPbPbCen[010,3050]data1_[prob_value]_[18r,18q].root
 2) output file (containig h_raw_signal_prob[prob], h_invmass_[ptmin]_[ptmax]_prob[prob], h_invmasstot_[ptmin]_[ptmax]_prob[prob]):
    raw_yields_[010,3050].root
 
 .L merge_and_fit_histo.C
 merge_and_fit_histo()
 */

#include "AliHFInvMassFitter.h"


Double_t mass = 2.286;
Double_t min_fit = mass-0.14;
Double_t max_fit = mass+0.14;
Int_t signal_fit_f = 0;    // kGaus=0, k2Gaus=1, k2GausSigmaRatioPar=2
Int_t background_fit_f = 2;// kExpo=0, kLin=1, kPol2=2, kNoBk=3, kPow=4, kPowEx=5
Int_t rebin = 5;

const int nptbins = 2;
float ptlimits[nptbins+1] = {6, 8, 12};
TString dir[nptbins] = {"6-8", "8-12"};

const int nprob_test = 3;
TString   prob_test[nprob_test] = {"0.80", "0.85", "0.90"};

const int nfiles = 2;
TString   files[nfiles] = {"18q", "18r"};
TString   cent = "010";

bool fixsigma = true;
float sigmafix[nptbins] = {0.013, 0.013};

void merge_and_fit_invmasshisto(){
    
    TFile *fout = new TFile(Form("raw_yields_%s.root",cent.Data()),"RECREATE");
    TH1F* h_raw_signal[nprob_test];
    for(int k=0; k<nprob_test; k++){
        h_raw_signal[k] = new TH1F(Form("h_raw_signal_prob%s",prob_test[k].Data()),";#it{p}_{T} (GeV/#it{c});raw yield",nptbins,ptlimits);
    }
        
    
    for(int i=0; i<nptbins; i++){
        TCanvas *c_fit = new TCanvas(Form("fits_%.0f_%.0f",ptlimits[i],ptlimits[i+1]),Form("fits_%.0f_%.0f",ptlimits[i],ptlimits[i+1]));
        c_fit->Divide(nprob_test,2);
        
        for(int iprob=0; iprob<nprob_test; iprob++){
            TH1F *hmerge=0x0;
            TCanvas *c_fit_prob = new TCanvas(Form("fits_prob%s_%.0f_%.0f",prob_test[iprob].Data(),ptlimits[i],ptlimits[i+1]),Form("fits_prob%s_%.0f_%.0f",prob_test[iprob].Data(),ptlimits[i],ptlimits[i+1]));
            c_fit_prob->Divide(3,2);
            // single period analysis
            for(int ifil=0; ifil<nfiles; ifil++){
                TFile *f = TFile::Open(Form("%s/masshistoLctopK0sPbPbCen%sdata1_%s_%s.root",dir[i].Data(),cent.Data(),prob_test[iprob].Data(),files[ifil].Data()));
                TH1F *h = (TH1F*)f->Get(Form("h_invmass%.0f_%.0f_prob%s",ptlimits[i],ptlimits[i+1],prob_test[iprob].Data()));
                h->SetName(Form("h_invmass%s_%.0f_%.0f_prob%s",files[ifil].Data(),ptlimits[i],ptlimits[i+1],prob_test[iprob].Data()));
                if(ifil==0){
                    hmerge=(TH1F*)h->Clone("h_merge");
                    hmerge->SetName(Form("h_invmasstot_%.0f_%.0f_prob%s",ptlimits[i],ptlimits[i+1],prob_test[iprob].Data()));
                }
                else{
                    hmerge->Add(h);
                }
                c_fit_prob->cd(ifil+1);
                gPad->SetTicks();
                h->Rebin(rebin);
                Float_t bin_width = h->GetXaxis()->GetBinWidth(3);
                TString histo_title=Form("%.0f-%.0f prob>%s %s; #it{M} (GeV/#it{c}^{2}); Counts/%.0f MeV/#it{c}^{2}",ptlimits[i],ptlimits[i+1],prob_test[iprob].Data(),files[ifil].Data(),bin_width*1000.);
                h->SetTitle(histo_title.Data());
                AliHFInvMassFitter *fitter = new AliHFInvMassFitter(h,min_fit,max_fit,background_fit_f,signal_fit_f);
                fitter->SetUseLikelihoodFit();
                fitter->SetInitialGaussianMean(mass);
                if(fixsigma)fitter->SetFixGaussianSigma(sigmafix[i]);
                Bool_t out=fitter->MassFitter(0);
                if(!out) {
                    fitter->GetHistoClone()->Draw();
                }
                fitter->DrawHere(gPad);
                Double_t sigma=fitter->GetSigma();
                fout->cd();
                h->Write();
                
                // subtracting background
                TF1 *bkgf = (TF1*)fitter->GetBackgroundFullRangeFunc();
                TH1F *hbkg = (TH1F*)h->Clone("hbkg");
                TH1F *hsigsub = (TH1F*)h->Clone("hsigsub");
                for(int j=0; j<h->GetNbinsX(); j++){
                    float bkg=bkgf->Eval(hbkg->GetBinCenter(j+1));
                    float bkge=TMath::Sqrt(bkg);
                    hbkg->SetBinContent(j+1,bkg);
                    hbkg->SetBinError(j+1,bkge);
                }
                //hbkg->Draw("same");
                hsigsub->Add(hbkg,-1.);
                c_fit_prob->cd(ifil+4);
                AliHFInvMassFitter *fitterbkg = new AliHFInvMassFitter(hsigsub,min_fit,max_fit,3,signal_fit_f);
                fitterbkg->SetUseChi2Fit();
                fitterbkg->SetInitialGaussianMean(mass);
                fitterbkg->SetFixGaussianSigma(sigma);
                Bool_t out2=fitterbkg->MassFitter(0);
                if(!out2) {
                    fitterbkg->GetHistoClone()->Draw();
                }
                fitterbkg->DrawHere(gPad);
            }
            // merging 2 periods
            hmerge->Rebin(rebin);
            Float_t bin_width = hmerge->GetXaxis()->GetBinWidth(3);
            TString histo_title=Form("%.0f-%.0f prob>%s; #it{M} (GeV/#it{c}^{2}); Counts/%.0f MeV/#it{c}^{2}",ptlimits[i],ptlimits[i+1],prob_test[iprob].Data(),bin_width*1000.);
            hmerge->SetTitle(histo_title.Data());
            c_fit->cd(iprob+1);
            gPad->SetTicks();
            AliHFInvMassFitter *fitter = new AliHFInvMassFitter(hmerge,min_fit,max_fit,background_fit_f,signal_fit_f);
            fitter->SetUseLikelihoodFit();
            fitter->SetInitialGaussianMean(mass);
            if(fixsigma)fitter->SetFixGaussianSigma(sigmafix[i]);
            Bool_t out=fitter->MassFitter(0);
            if(!out) {
                fitter->GetHistoClone()->Draw();
            }
            fitter->DrawHere(gPad);
            c_fit_prob->cd(3);
            fitter->DrawHere(gPad);
            Double_t sigma=fitter->GetSigma();
            double rawyield = fitter->GetRawYield();
            double rawyielderr = fitter->GetRawYieldError();
            h_raw_signal[iprob]->SetBinContent(i+1,rawyield);
            h_raw_signal[iprob]->SetBinError(i+1,rawyielderr);
            
            //h_signal
            
            fout->cd();
            hmerge->Write();
            
            // subtracting background
            TF1 *bkgf = (TF1*)fitter->GetBackgroundFullRangeFunc();
            TH1F *hbkg = (TH1F*)hmerge->Clone("hbkg");
            TH1F *hsigsub = (TH1F*)hmerge->Clone("hsigsub");
            for(int j=0; j<hmerge->GetNbinsX(); j++){
                float bkg=bkgf->Eval(hbkg->GetBinCenter(j+1));
                float bkge=TMath::Sqrt(bkg);
                hbkg->SetBinContent(j+1,bkg);
                hbkg->SetBinError(j+1,bkge);
            }
            hsigsub->Add(hbkg,-1.);
            c_fit_prob->cd(6);
            AliHFInvMassFitter *fitterbkg = new AliHFInvMassFitter(hsigsub,min_fit,max_fit,3,signal_fit_f);
            //fitterbkg->SetUseLikelihoodFit();
            fitterbkg->SetUseChi2Fit();
            fitterbkg->SetInitialGaussianMean(mass);
            fitterbkg->SetFixGaussianSigma(sigma);
            Bool_t out2=fitterbkg->MassFitter(0);
            if(!out2) {
                fitterbkg->GetHistoClone()->Draw();
            }
            fitterbkg->DrawHere(gPad);
            c_fit->cd(iprob+4);
            fitterbkg->DrawHere(gPad);
        }
    }
    fout->cd();
    for(int k=0; k<nprob_test; k++){
        h_raw_signal[k]->Write();
    }
}

