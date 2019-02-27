#include "AliHFInvMassFitter.h"


const int nfiles=1;
TString filename[nfiles] = {"histototal.root"};
TString histo_name = {"h_invmass_"};
Float_t ptlimits[nfiles+1] = {1, 2};
Double_t mass = 1.864;
Double_t min_fit = mass-0.14;
Double_t max_fit = mass+0.14;
Int_t signal_fit_f = 0; // kGaus=0, k2Gaus=1, k2GausSigmaRatioPar=2
Int_t background_fit_f = 0;// kExpo=0, kLin=1, kPol2=2, kNoBk=3, kPow=4, kPowEx=5
Int_t rebin = 1;


void mass_fitter(){
    
    for(int i=0; i<nfiles; i++){
        TFile *f = TFile::Open(filename[i].Data());
        f->ls();
        histo_name += Form("%.0f-%.0f",ptlimits[i],ptlimits[i+1]);
        TH1F  *h = (TH1F*)f->Get(histo_name.Data());
        if(!h) Printf("ERROR: no histo: %s",histo_name.Data());
        h->Rebin(rebin);
        Float_t bin_width = h->GetXaxis()->GetBinWidth(3);
        TString histo_title=Form("%.0f-%.0f; #it{M} (GeV/#it{c}^{2}); Counts/%.0f MeV/#it{c}^{2}",ptlimits[i],ptlimits[i+1],bin_width*1000.);
        h->SetTitle(histo_title.Data());
        
        TCanvas *c_fit = new TCanvas(Form("Fit_%.0f-%.0f",ptlimits[i],ptlimits[i+1]),Form("Fit_%.0f-%.0f",ptlimits[i],ptlimits[i+1]));
        c_fit->cd();
        gPad->SetTicks();
        AliHFInvMassFitter *fitter = new AliHFInvMassFitter(h,min_fit,max_fit,background_fit_f,signal_fit_f);
        fitter->SetUseLikelihoodFit();
        fitter->SetInitialGaussianMean(mass);
        //fitter->SetFixGaussianSigma(0.014);
        Bool_t out=fitter->MassFitter(0);
        if(!out) {
            fitter->GetHistoClone()->Draw();
        }
        fitter->DrawHere(gPad);
    }
    
 
}
