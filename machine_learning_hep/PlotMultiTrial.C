
void PlotMultiTrial(const char* filepath, double rawYieldRef, double meanRef, double sigmaRef, double chi2Ref, double chi2Cut, Bool_t* usedBkgs, Int_t nBinsBincount, Bool_t considerFreeSigma, const char* saveDir, const char* suffix, const char* title, TDirectory* derivedResultsDir) {

    // Extract the trials
    TString esesel(saveDir);
    TString bkgTreat("");
    TFile* inputFile=new TFile(filepath, "READ");
    
    std::vector<Int_t> bkgColors = {kPink - 6, kCyan + 3, kGreen - 1, kYellow - 2, kRed - 6, kBlue - 6};
    const Int_t nBackFuncCases=6;
    const Int_t nConfigCases=6;
    Int_t colorBC0=kGreen+2;
    Int_t colorBC1=kOrange+5;
    Int_t minBCrange=1;
    Int_t maxBCrange=nBinsBincount;
    Int_t nBCranges=nBinsBincount;
    TString confCase[nConfigCases]={"FixedSigFreeMean","FixedSigUp","FixedSigDw","FreeSigFreeMean","FreeSigFixedMean","FixedSigFixedMean"};
    TString bkgFunc[nBackFuncCases]={"Expo","Lin","Pol2","Pol3","Pol4","Pol5"};
    
    const Int_t totCases=nConfigCases*nBackFuncCases;
    
    // 0 => not used; 1 => used for fit; 2 => used also for bin count
    Int_t mask[totCases]={0,0,0,0,0,0,   // fixed sigma, free mean (Expo, Lin, Pol2,Pol3,Pol4)
                          0,0,0,0,0,0,   // fixed sigma upper
                          0,0,0,0,0,0,   // fixed sigma lower
                          0,0,0,0,0,0,   // free sigma, free mean
                          0,0,0,0,0,0,   // free sigma, fixed mean
                          0,0,0,0,0,0,   // fixed mean, fixed sigma
    };
    // Enable only the background cases we ran the multi trial with
    Int_t plotCase = (maxBCrange >= minBCrange) ? 2 : 1;
    for(Int_t i = 0; i < 6; i++) {
        mask[i] = (usedBkgs[i] > 0) ? plotCase : 0;
        mask[30+i] = (usedBkgs[i] > 0) ? plotCase : 0;
        if(considerFreeSigma) {
            mask[18+i] = (usedBkgs[i] > 0) ? plotCase : 0;
            mask[24+i] = (usedBkgs[i] > 0) ? plotCase : 0;
        }
    }
    
    TH1F* histo6[totCases];
    cout << "nconfigcases " << nConfigCases << "\t nbackgfunccases " << nBackFuncCases << endl;
    Int_t kjh=0;
    for(Int_t iConf=0; iConf<nConfigCases; iConf++){
        for(Int_t iType=0; iType<nBackFuncCases; iType++){
            histo6[kjh++]=(TH1F*)inputFile->Get(Form("hRawYieldTrial%s%s%s",bkgFunc[iType].Data(),confCase[iConf].Data(),bkgTreat.Data()));
            if (!histo6[kjh]) cout << "Histo6 " << Form("hRawYieldTrial%s%s%s",bkgFunc[iType].Data(),confCase[iConf].Data(),bkgTreat.Data()) << " not found " << endl;
        }
    }
    
    Int_t totTrials=0;
    Int_t totTrialsBC0=0;
    Int_t totTrialsBC1=0;
    Int_t totHistos=0;
    Int_t first[totCases];
    Int_t last[totCases];
    Int_t firstBC0[totCases];
    Int_t lastBC0[totCases];
    Int_t firstBC1[totCases];
    Int_t lastBC1[totCases];
    TLine **vlines=new TLine*[totCases];
    TLatex **tlabels=new TLatex*[totCases+1];

    for(Int_t nc=0; nc<totCases; nc++) {
        if(mask[nc] == 0 || !histo6[nc]) {
            mask[nc]=0;
            continue;
        }
        first[nc]=totTrials;
        totTrials+=histo6[nc]->GetNbinsX();
        last[nc]=totTrials;
        ++totHistos;

        printf("  %d) %s  -- %d \n",nc,histo6[nc]->GetName(),first[nc]);
        vlines[nc]=new TLine(last[nc],0.,last[nc],50000.);
        vlines[nc]->SetLineColor(kMagenta+2);
        vlines[nc]->SetLineStyle(2);
        TString ttt=histo6[nc]->GetName();
        ttt.ReplaceAll("hRawYieldTrial","");
        if(ttt.Contains("FixedMean")) ttt="Fix #mu";
        if(ttt.Contains("FixedSp20")) ttt="#sigma+";
        if(ttt.Contains("FixedSm20")) ttt="#sigma-";
        if(ttt.Contains("FreeS")) ttt="Free #sigma";
        ttt.ReplaceAll("FixedS","");
        if(bkgTreat != "" && ttt.Contains(bkgTreat.Data())) {
            ttt.ReplaceAll(bkgTreat.Data(),"");
        }
        tlabels[nc]=new TLatex(first[nc]+0.02*totTrials,10,ttt.Data());
        tlabels[nc]->SetTextColor(kMagenta+2);
        tlabels[nc]->SetTextColor(kMagenta+2);

        if(mask[nc]==2) {
            TString hbcname=histo6[nc]->GetName();
            // Take bin count from background function of total fit
            hbcname.ReplaceAll("Trial","TrialBinC0");
            cout<< " name bc " << hbcname.Data() << endl;
            TH2F* hbc2dt=(TH2F*)inputFile->Get(hbcname.Data());
            firstBC0[nc]=totTrialsBC0;
            totTrialsBC0+=hbc2dt->GetNbinsX();
            lastBC0[nc]=totTrialsBC0;

            hbcname.ReplaceAll("TrialBinC0","TrialBinC1");
            cout<< " name bc " << hbcname.Data() << endl;
            hbc2dt=(TH2F*)inputFile->Get(hbcname.Data());
            firstBC1[nc]=totTrialsBC1;
            totTrialsBC1+=hbc2dt->GetNbinsX();
            lastBC1[nc]=totTrialsBC1;
        }
    }
    
    tlabels[totCases]=new TLatex(totTrials+30,10,"BinCnt");
    tlabels[totCases]->SetTextColor(kMagenta+2);
    tlabels[totCases]->SetTextColor(kMagenta+2);
    
    printf("Histos merged = %d    totTrials=%d\n",totHistos,totTrials);
    
    Printf("tottrials \t tottrialsBC0 \t  nBC ranges \t ");
    Printf("%d \t %d \t %d",totTrials,totTrialsBC0,nBCranges);

    // We need one histogram per background function, do it brute force with a std::map
    // Bin counts will be just summarised for now in one histogram
    std::map<TString, TH1*> hRawYieldAllBkgs;
    std::map<TString, TH1*> hMeanAllBkgs;
    std::map<TString, TH1*> hSigmaAllBkgs;
    std::map<TString, TH1*> hChi2AllBkgs;
    for(Int_t nc=0; nc<totCases; nc++){
        if(mask[nc]){
            TString hmeanname=histo6[nc]->GetName();
            for(Int_t iBkg = 0; iBkg < nBackFuncCases; iBkg++) {
                auto it = hRawYieldAllBkgs.find(bkgFunc[iBkg]);
                if( it != hRawYieldAllBkgs.end()) {
                    continue;
                }
                if(hmeanname.Contains(bkgFunc[iBkg])) {
                    hRawYieldAllBkgs[bkgFunc[iBkg]] = new TH1F(Form("hRawYieldAll_%s_%s", bkgFunc[iBkg].Data(), suffix),
                                                               " ; Trial # ; raw yield", totTrials, 0., totTrials);
                    hRawYieldAllBkgs[bkgFunc[iBkg]]->SetLineColor(bkgColors[iBkg]);
                    hRawYieldAllBkgs[bkgFunc[iBkg]]->SetMarkerColor(bkgColors[iBkg]);
                    hRawYieldAllBkgs[bkgFunc[iBkg]]->SetStats(0);

                    hMeanAllBkgs[bkgFunc[iBkg]] = new TH1F(Form("hMeanAll_%s_%s", bkgFunc[iBkg].Data(), suffix),
                                                               " ; Trial # ; Gaussian mean", totTrials, 0., totTrials);
                    hMeanAllBkgs[bkgFunc[iBkg]]->SetLineColor(bkgColors[iBkg]);
                    hMeanAllBkgs[bkgFunc[iBkg]]->SetMarkerColor(bkgColors[iBkg]);
                    hMeanAllBkgs[bkgFunc[iBkg]]->SetMinimum(0.8 * meanRef);
                    hMeanAllBkgs[bkgFunc[iBkg]]->SetMaximum(1.2 * meanRef);
                    hMeanAllBkgs[bkgFunc[iBkg]]->SetStats(0);

                    hSigmaAllBkgs[bkgFunc[iBkg]] = new TH1F(Form("hSigmaAll_%s_%s", bkgFunc[iBkg].Data(), suffix),
                                                               " ; Trial # ; Gaussian Sigma", totTrials, 0., totTrials);
                    hSigmaAllBkgs[bkgFunc[iBkg]]->SetLineColor(bkgColors[iBkg]);
                    hSigmaAllBkgs[bkgFunc[iBkg]]->SetMarkerColor(bkgColors[iBkg]);
                    hSigmaAllBkgs[bkgFunc[iBkg]]->SetMinimum(0.);
                    hSigmaAllBkgs[bkgFunc[iBkg]]->SetMaximum(1.1 * sigmaRef);
                    hSigmaAllBkgs[bkgFunc[iBkg]]->SetStats(0);

                    hChi2AllBkgs[bkgFunc[iBkg]] = new TH1F(Form("hChi2All_%s_%s", bkgFunc[iBkg].Data(), suffix),
                                                               " ; Trial # ; Gaussian Sigma", totTrials, 0., totTrials);
                    hChi2AllBkgs[bkgFunc[iBkg]]->SetLineColor(bkgColors[iBkg]);
                    hChi2AllBkgs[bkgFunc[iBkg]]->SetMarkerColor(bkgColors[iBkg]);
                    hChi2AllBkgs[bkgFunc[iBkg]]->SetMarkerStyle(7);
                    hChi2AllBkgs[bkgFunc[iBkg]]->SetStats(0);
                }
            }
        }
    }

    TH1F* hRawYieldAllBC0=new TH1F(Form("hRawYieldAllBC0_%s", suffix),
                                   " ; Trial # ; raw yield",
                                   totTrialsBC0*nBCranges,
                                   0.,
                                   totTrialsBC0*nBCranges);

    TH1F* hRawYieldAllBC1=new TH1F(Form("hRawYieldAllBC1_%s", suffix),
                                   " ; Trial # ; raw yield",
                                   totTrialsBC1*nBCranges,
                                   0.,
                                   totTrialsBC1*nBCranges);

    
    Double_t lowerEdgeYieldHistos = rawYieldRef - 1.5 * rawYieldRef;
    if(lowerEdgeYieldHistos < 0) {
        lowerEdgeYieldHistos = 0;
    }
    Double_t upperEdgeYieldHistos = rawYieldRef + 1.5 * rawYieldRef;
    TH1F* hRawYieldDistAll=new TH1F("hRawYieldDistAll","  ; raw yield",200,lowerEdgeYieldHistos,upperEdgeYieldHistos);
    hRawYieldDistAll->SetFillStyle(3003);
    hRawYieldDistAll->SetFillColor(kBlue+1);
    TH1F* hRawYieldDistAllBC0=new TH1F("hRawYieldDistAllBC0","  ; raw yield",200,lowerEdgeYieldHistos,upperEdgeYieldHistos);
    TH1F* hRawYieldDistAllBC1=new TH1F("hRawYieldDistAllBC1","  ; raw yield",200,lowerEdgeYieldHistos,upperEdgeYieldHistos);
    hRawYieldDistAllBC0->SetFillStyle(3004);
    hRawYieldDistAllBC1->SetFillStyle(3004);
    // NOTE Note uses at the moment
    //TH1F* hStatErrDistAll=new TH1F("hStatErrDistAll","  ; Stat Unc on Yield",300,0,10000);
    //TH1F* hRelStatErrDistAll=new TH1F("hRelStatErrDistAll","  ; Rel Stat Unc on Yield",100,0.,1.);
    
    Double_t minYield=999999.;
    Double_t maxYield=0.;
    Double_t sumy[4]={0.,0.,0.,0.};
    Double_t sumwei[4]={0.,0.,0.,0.};
    Double_t sumerr[4]={0.,0.,0.,0.};
    Double_t counts=0.;
    Double_t wei[4];
    Double_t maxFilled=-1;
    Printf("trial \t first \t last \t firstBC0 \t lastBC0 ");
    for(Int_t nc=0; nc<totCases; nc++){
        if(mask[nc]){
            Printf("%d \t %d \t %d \t %d \t %d",nc,first[nc],last[nc],firstBC0[nc],lastBC0[nc]);
            TString hmeanname=histo6[nc]->GetName();
            hmeanname.ReplaceAll("RawYield","Mean");
            TH1F* hmeant6=(TH1F*)inputFile->Get(hmeanname.Data());
            
            TString hsigmaname=histo6[nc]->GetName();
            hsigmaname.ReplaceAll("RawYield","Sigma");
            TH1F* hsigmat6=(TH1F*)inputFile->Get(hsigmaname.Data());
            
            TString hchi2name=histo6[nc]->GetName();
            hchi2name.ReplaceAll("RawYield","Chi2");
            TH1F* hchi2t6=(TH1F*)inputFile->Get(hchi2name.Data());
            
            TString hbcname=histo6[nc]->GetName();
            hbcname.ReplaceAll("Trial","TrialBinC0");
            TH2F* hbc2dt060=(TH2F*)inputFile->Get(hbcname.Data());
            hbcname.ReplaceAll("TrialBinC0","TrialBinC1");
            TH2F* hbc2dt060_bc1=(TH2F*)inputFile->Get(hbcname.Data());
            for(Int_t ib=1; ib<=histo6[nc]->GetNbinsX(); ib++){
                Double_t ry=histo6[nc]->GetBinContent(ib);
                //cout<< " ry " << ry <<endl;
                Double_t ery=histo6[nc]->GetBinError(ib);
                
                Double_t pos=hmeant6->GetBinContent(ib);
                Double_t epos=hmeant6->GetBinError(ib);
                
                Double_t sig=hsigmat6->GetBinContent(ib);
                Double_t esig=hsigmat6->GetBinError(ib);
                
                Double_t chi2=hchi2t6->GetBinContent(ib);

                // Fill 
                if(ry>0.001 && ery>(0.01*ry) && ery<(0.5*ry) && chi2<chi2Cut){
                // This also throws away some chi2 == 0
                //if(chi2<chi2Cut && chi2>0){

                    // Get the right histograms to fill
                    TString bkgFuncName = hmeant6->GetName();
                    for(Int_t iBkg = 0; iBkg < nBackFuncCases; iBkg++) {
                        if(bkgFuncName.Contains(bkgFunc[iBkg])) {
                            bkgFuncName = bkgFunc[iBkg];
                            break;
                        }
                    }
                    hRawYieldDistAll->Fill(ry);
                    hRawYieldAllBkgs[bkgFuncName]->SetBinContent(first[nc]+ib,ry);
                    hRawYieldAllBkgs[bkgFuncName]->SetBinError(first[nc]+ib,ery);
                    // NOTE Not used at the moment
                    //hStatErrDistAll->Fill(ery);
                    //hRelStatErrDistAll->Fill(ery/ry);
                    if(ry<minYield) minYield=ry;
                    if(ry>maxYield) maxYield=ry;
                    wei[0]=1.;
                    wei[1]=1./(ery*ery);
                    wei[2]=1./(ery*ery/(ry*ry));
                    wei[3]=1./(ery*ery/ry);
                    for(Int_t kw=0; kw<4; kw++){
                        sumy[kw]+=wei[kw]*ry;
                        sumerr[kw]+=wei[kw]*wei[kw]*ery*ery;
                        sumwei[kw]+=wei[kw];
                    }
                    counts+=1.;
                    hSigmaAllBkgs[bkgFuncName]->SetBinContent(first[nc]+ib,sig);
                    hSigmaAllBkgs[bkgFuncName]->SetBinError(first[nc]+ib,esig);
                    hMeanAllBkgs[bkgFuncName]->SetBinContent(first[nc]+ib,pos);
                    hMeanAllBkgs[bkgFuncName]->SetBinError(first[nc]+ib,epos);
                    hChi2AllBkgs[bkgFuncName]->SetBinContent(first[nc]+ib,chi2);
                    hChi2AllBkgs[bkgFuncName]->SetBinError(first[nc]+ib,0.000001);
                    if(mask[nc]==2){
                        for(Int_t iy=minBCrange; iy<=maxBCrange;iy++){
                            std::cout << "####### BC #######" << std::endl;
                            Double_t bc=hbc2dt060->GetBinContent(ib,iy);
                            Double_t ebc=hbc2dt060->GetBinError(ib,iy);
                            Double_t bc_1=hbc2dt060_bc1->GetBinContent(ib,iy);
                            Double_t ebc_1=hbc2dt060_bc1->GetBinError(ib,iy);
                            //if(bc>0.001 && ebc<0.5*bc && bc<5.*ry){
                            std::cout << bc << std::endl;
                            if(bc>0.001){
                                Int_t theBin=iy+(firstBC0[nc]-1)*nBCranges;
                                cout<< " bin content " << bc << " the bin " << theBin << " BCrange " << iy << endl;
                                hRawYieldAllBC0->SetBinContent(theBin-2,bc);
                                hRawYieldAllBC0->SetBinError(theBin-2,ebc);
                                hRawYieldDistAllBC0->Fill(bc);
                                if(hRawYieldAllBC0->GetBinCenter(theBin-2)>maxFilled) maxFilled=hRawYieldAllBC0->GetBinCenter(theBin-2);
                                theBin=iy+(firstBC1[nc]-1)*nBCranges;
                                hRawYieldAllBC1->SetBinContent(theBin-2,bc_1);
                                hRawYieldAllBC1->SetBinError(theBin-2,ebc_1);
                                hRawYieldDistAllBC1->Fill(bc_1);
                                if(hRawYieldAllBC1->GetBinCenter(theBin-2)>maxFilled) maxFilled=hRawYieldAllBC1->GetBinCenter(theBin-2);
                            }
                        }
                    }
                }
            }
        }
    }
    
    Double_t weiav[4]={0.,0.,0.,0.};
    Double_t eweiav[4]={0.,0.,0.,0.};
    for(Int_t kw=0; kw<4; kw++){
        if(sumwei[kw]>0.){
            weiav[kw]=sumy[kw]/sumwei[kw];
            eweiav[kw]=TMath::Sqrt(sumerr[kw])/sumwei[kw];
        }
    }
    

    hRawYieldAllBC0->SetStats(0);
    hRawYieldAllBC0->SetMarkerColor(colorBC0);
    hRawYieldAllBC0->SetLineColor(colorBC0);
    hRawYieldDistAllBC0->SetLineColor(colorBC0);
    hRawYieldDistAllBC0->SetFillColor(colorBC0);
    hRawYieldDistAllBC0->SetLineWidth(1);
    hRawYieldDistAllBC0->SetLineStyle(1);
    hRawYieldDistAllBC0->Scale(hRawYieldDistAll->GetEntries()/hRawYieldDistAllBC0->GetEntries());

    hRawYieldAllBC1->SetStats(0);
    hRawYieldAllBC1->SetMarkerColor(colorBC1);
    hRawYieldAllBC1->SetLineColor(colorBC1);
    hRawYieldDistAllBC1->SetLineColor(colorBC1);
    hRawYieldDistAllBC1->SetFillColor(colorBC1);
    hRawYieldDistAllBC1->SetLineWidth(1);
    hRawYieldDistAllBC1->SetLineStyle(1);
    hRawYieldDistAllBC1->Scale(hRawYieldDistAll->GetEntries()/hRawYieldDistAllBC1->GetEntries());

    hRawYieldDistAll->SetStats(0);
    hRawYieldDistAll->SetLineWidth(1);

    // Write fit and bin count distribution for further usage
    derivedResultsDir->WriteObject(hRawYieldDistAll, "h_mt_fit");
    derivedResultsDir->WriteObject(hRawYieldDistAllBC0, "h_mt_bc0");
    derivedResultsDir->WriteObject(hRawYieldDistAllBC1, "h_mt_bc1");
    
    TLine *l=new TLine(rawYieldRef,0.,rawYieldRef,hRawYieldDistAll->GetMaximum());
    l->SetLineColor(kRed);
    l->SetLineWidth(2);
    
    TLine *ll=new TLine(0.,rawYieldRef,totTrials,rawYieldRef);
    ll->SetLineColor(kRed);
    ll->SetLineWidth(2);

    TCanvas* call=new TCanvas(Form("canvas_%s_6pad", suffix), "All",1400,800);
    call->Divide(3,2);
    call->cd(1);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    for(auto& h : hSigmaAllBkgs) {
        h.second->GetYaxis()->SetTitleOffset(1.7);
        h.second->Draw("same");
    }
    call->cd(2);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    for(auto& h : hMeanAllBkgs) {
        h.second->GetYaxis()->SetTitleOffset(1.7);
        h.second->Draw("same");
    }
    call->cd(3);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    for(auto& h : hChi2AllBkgs) {
        h.second->GetYaxis()->SetTitleOffset(1.7);
        h.second->Draw("same");
    }
    call->cd(4);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    Double_t newmax= 0.;
    for(auto& h : hRawYieldAllBkgs) {
        auto tmpMax =  1.25*(h.second->GetMaximum()+h.second->GetBinError(1));
        if(tmpMax > newmax) {
            newmax = tmpMax;
        }
        h.second->GetYaxis()->SetTitleOffset(1.7);
    }
   
    for(auto& h : hRawYieldAllBkgs) {
        if(maxFilled>0) h.second->GetXaxis()->SetRangeUser(0.,maxFilled);
        h.second->SetMaximum(newmax);
        h.second->SetTitle(title);
        h.second->Draw("same");
    }
    //hRawYieldAllBC0->Draw("same");
    //hRawYieldAllBC1->Draw("same");
    ll->Draw("same");
    /*
    TLatex* tweimean[4];
    for(Int_t kw=0; kw<4; kw++){
        tweimean[kw]=new TLatex(0.16,0.84-0.06*kw,Form("<Yield>_{wei%d} = %.1f #pm %.1f\n",kw,weiav[kw],eweiav[kw]*sqrt(counts)));
        tweimean[kw]->SetNDC();
        tweimean[kw]->SetTextColor(4);
        //    tweimean[kw]->Draw();
    }
    
    for(Int_t j=0; j<totCases; j++){
        if(mask[j]){
            vlines[j]->SetY2(newmax);
            vlines[j]->Draw("same");
            tlabels[j]->SetY(0.05*newmax);
            tlabels[j]->Draw();
        }
    }
    tlabels[totCases]->SetY(0.05*newmax);
    tlabels[totCases]->Draw();
    */
    call->cd(5);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.06);
    hRawYieldDistAll->SetTitle(title);
    hRawYieldDistAll->Draw();
    hRawYieldDistAll->GetXaxis()->SetRangeUser(minYield*0.8,maxYield*1.2);
    hRawYieldDistAllBC0->Draw("sameshist");
    hRawYieldDistAllBC1->Draw("sameshist");
    l->Draw("same");
    gPad->Update();
    // TODO This needs to be taken care of. At least for pK0s it does not work but it doesn't seem to be important as it's not used
    //TPaveStats* st=(TPaveStats*)hRawYieldDistAll->GetListOfFunctions()->FindObject("stats");
    //st->SetY1NDC(0.71);
    //st->SetY2NDC(0.9);
    //TPaveStats* stb0=(TPaveStats*)hRawYieldDistAllBC0->GetListOfFunctions()->FindObject("stats");
    //stb0->SetY1NDC(0.51);
    //stb0->SetY2NDC(0.7);
    //stb0->SetTextColor(hRawYieldDistAllBC0->GetLineColor());
    Double_t perc[3]={0.15,0.5,0.85}; // quantiles for +-1 sigma
    Double_t lim70[3];
    hRawYieldDistAll->GetQuantiles(3,lim70,perc);
    call->cd(6);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.06);
    Double_t aver=hRawYieldDistAll->GetMean();
    TLatex* tmean=new TLatex(0.15,0.93,Form("mean=%.3f",aver));
    tmean->SetNDC();
    tmean->Draw();
    TLatex* tmedian=new TLatex(0.15,0.86,Form("median=%.3f",lim70[1]));
    tmedian->SetNDC();
    tmedian->Draw();
    Double_t averBC0=hRawYieldDistAllBC0->GetMean();
    TLatex* tmeanBC0=new TLatex(0.15,0.79,Form("mean(BinCount0)=%.3f",averBC0));
    tmeanBC0->SetNDC();
    tmeanBC0->SetTextColor(hRawYieldDistAllBC0->GetLineColor());
    tmeanBC0->Draw();
    Double_t averBC1=hRawYieldDistAllBC1->GetMean();
    TLatex* tmeanBC1=new TLatex(0.15,0.72,Form("mean(BinCount1)=%.3f",averBC1));
    tmeanBC1->SetNDC();
    tmeanBC1->SetTextColor(hRawYieldDistAllBC1->GetLineColor());
    tmeanBC1->Draw();
    Double_t val=hRawYieldDistAll->GetRMS();
    TLatex* thrms=new TLatex(0.15,0.62,Form("rms=%.3f  (%.2f%%)",val,val/aver*100.));
    thrms->SetNDC();
    thrms->Draw();
    val=hRawYieldDistAllBC0->GetRMS();
    TLatex* thrmsBC0=new TLatex(0.15,0.55,Form("rms(BinCount0)=%.3f  (%.2f%%)",val,val/averBC0*100.));
    thrmsBC0->SetNDC();
    thrmsBC0->SetTextColor(hRawYieldDistAllBC0->GetLineColor());
    thrmsBC0->Draw();
    val=hRawYieldDistAllBC1->GetRMS();
    TLatex* thrmsBC1=new TLatex(0.15,0.48,Form("rms(BinCount1)=%.3f  (%.2f%%)",val,val/averBC1*100.));
    thrmsBC1->SetNDC();
    thrmsBC1->SetTextColor(hRawYieldDistAllBC1->GetLineColor());
    thrmsBC1->Draw();
    TLatex* tmin=new TLatex(0.15,0.38,Form("min=%.3f      max=%.2f",minYield,maxYield));
    tmin->SetNDC();
    tmin->Draw();
    val=(maxYield-minYield)/sqrt(12);
    TLatex* trms=new TLatex(0.15,0.31,Form("(max-min)/sqrt(12)=%.3f  (%.2f%%)",val,val/aver*100.));
    trms->SetNDC();
    trms->Draw();
    TLatex* meanRefLabel=new TLatex(0.15,0.24,Form("mean(ref)=%.3f",rawYieldRef));
    meanRefLabel->SetNDC();
    meanRefLabel->SetTextColor(kRed);
    meanRefLabel->Draw();
    //
    TLatex* meanRefDiff=new TLatex(0.15,0.17,Form("mean(ref)-mean(fit)=%.3f  (%.2f%%)",rawYieldRef-aver,100.*(rawYieldRef-aver)/rawYieldRef));
    meanRefDiff->SetNDC();
    meanRefDiff->SetTextColor(kBlack);
    meanRefDiff->Draw();
    TLatex* meanRefDiffBC0=new TLatex(0.15,0.10,Form("mean(ref)-mean(BC0)=%.3f  (%.2f%%)",rawYieldRef-averBC0,100.*(rawYieldRef-averBC0)/rawYieldRef));
    meanRefDiffBC0->SetNDC();
    meanRefDiffBC0->SetTextColor(hRawYieldDistAllBC0->GetLineColor());
    meanRefDiffBC0->Draw();
    TLatex* meanRefDiffBC1=new TLatex(0.15,0.03,Form("mean(ref)-mean(BC1)=%.3f  (%.2f%%)",rawYieldRef-averBC1,100.*(rawYieldRef-averBC1)/rawYieldRef));
    meanRefDiffBC1->SetNDC();
    meanRefDiffBC1->SetTextColor(hRawYieldDistAllBC1->GetLineColor());
    meanRefDiffBC1->Draw();
    //
    //
    //
    //
    //
    //
    //
    /*val=(maxYield-aver)/sqrt(3);
    TLatex* tup=new TLatex(0.15,0.39,Form("(max-mean)/sqrt(3)=%.3f  (%.2f%%)",val,val/aver*100.));
    tup->SetNDC();
    tup->Draw();
    val=(aver-minYield)/sqrt(3);
    TLatex* tdw=new TLatex(0.15,0.32,Form("(mean-min)/sqrt(3)=%.3f  (%.2f%%)",val,val/aver*100.));
    tdw->SetNDC();
    tdw->Draw();
    TLatex* tl15=new TLatex(0.15,0.22,Form("15 percentile=%.3f",lim70[0]));
    tl15->SetNDC();
    tl15->Draw();
    TLatex* tl85=new TLatex(0.15,0.13,Form("85 percentile=%.3f",lim70[2]));
    tl85->SetNDC();
    tl85->Draw();
    val=(lim70[2]-lim70[0])/2.;
    TLatex* t1s=new TLatex(0.15,0.06,Form("70%% range =%.3f  (%.2f%%)",val,val/aver*100.));
    t1s->SetNDC();
    t1s->Draw();*/
    call->SaveAs(Form("%s/6pad_pt_%s.eps", esesel.Data(), suffix));
    for(Int_t kw=0; kw<4; kw++){
        printf("Weight %d: %.1f +- %.1f(stat) +- %.1f (syst)\n",kw,weiav[kw],eweiav[kw]*sqrt(counts),(maxYield-minYield)/sqrt(12));
    }
    
    
}

