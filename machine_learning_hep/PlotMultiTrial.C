
void PlotMultiTrial(const char* filepath, double rawYieldRef, double meanRef, double sigmaRef, double chi2Ref, double chi2Cut, const char* saveDir, const char* suffix, const char* title) {

    // Extract the trials
    TString esesel(saveDir);
    TString bkgTreat("");
    TFile* fil6=new TFile(filepath, "READ");
    
    const Int_t nBackFuncCases=6;
    const Int_t nConfigCases=6;
    Int_t colorBC0=kGreen+2;
    Int_t minBCrange=3;
    Int_t maxBCrange=5;
    Int_t nBCranges=maxBCrange-minBCrange+1;
    TString confCase[nConfigCases]={"FixedSig","FixedSigUp","FixedSigDw","FreeSig","FixedMeanFreeSig","FixedMeanFixedSig"};
    TString bkgFunc[nBackFuncCases]={"Expo","Lin","Pol2","Pol3","Pol4","Pol5"};
    
    const Int_t totCases=nConfigCases*nBackFuncCases;
    
    // 0= not used; 1 = used for fit; 2= used also for bin count0, 3=use also bin count1, 4=use both binc
    Int_t mask[totCases]={0,0,4,4,0,0,   // fixed sigma (Expo, Lin, Pol2,Pol3,Pol4)
        0,0,0,0,0,0,   // fixed sigma upper
        0,0,0,0,0,0,   // fixed sigma lower
        0,0,4,4,0,0,   // free sigma, free mean
        0,0,0,0,0,0,   // free sigma, fixed mean
        0,0,0,0,0,0,   // fixed mean, fixed sigma
    };
    
    TH1F* histo6[totCases];
    cout << "nconfigcases " << nConfigCases << "\t nbackgfunccases " << nBackFuncCases << endl;
    Int_t kjh=0;
    for(Int_t iConf=0; iConf<nConfigCases; iConf++){
        for(Int_t iType=0; iType<nBackFuncCases; iType++){
            histo6[kjh++]=(TH1F*)fil6->Get(Form("hRawYieldTrial%s%s%s",bkgFunc[iType].Data(),confCase[iConf].Data(),bkgTreat.Data()));
            if (!histo6[kjh]) cout << "Histo6 " << Form("hRawYieldTrial%s%s%s",bkgFunc[iType].Data(),confCase[iConf].Data(),bkgTreat.Data()) << " not found " << endl;
        }
    }
    
    Int_t totTrials=0;
    Int_t totTrialsBC0=0;
    Int_t totHistos=0;
    Int_t first[totCases];
    Int_t last[totCases];
    Int_t firstBC0[totCases];
    Int_t lastBC0[totCases];
    Double_t minyd=9e9;
    Double_t maxyd=0;
    for(Int_t nc=0; nc<totCases; nc++){
        if(mask[nc]){
            //change all histo1 to histo6
            if(histo6[nc]){
                first[nc]=totTrials;
                totTrials+=histo6[nc]->GetNbinsX();
                last[nc]=totTrials;
                Double_t thisMin=histo6[nc]->GetBinContent(histo6[nc]->GetMinimumBin());
                Double_t thisMax=histo6[nc]->GetBinContent(histo6[nc]->GetMaximumBin());
                if(thisMin<minyd) minyd=thisMin;
                if(thisMax>maxyd) maxyd=thisMax;
                ++totHistos;
                if(mask[nc]==2 || mask[nc]==4){
                    TString hbcname=histo6[nc]->GetName();
                    //hbcname.ReplaceAll("Trial","TrialBinC0");
                    hbcname.ReplaceAll("Trial","TrialBinC0");
                    cout<< " name bc " << hbcname.Data() << endl;
                    TH2F* hbc2dt=(TH2F*)fil6->Get(hbcname.Data());
                    Int_t bnx,bny,bnz;
                    hbc2dt->GetMinimumBin(bnx,bny,bnz);
                    thisMin=hbc2dt->GetBinContent(bnx,bny);
                    hbc2dt->GetMaximumBin(bnx,bny,bnz);
                    thisMax=hbc2dt->GetBinContent(bnx,bny);
                    if(thisMin<minyd) minyd=thisMin;
                    if(thisMax>maxyd) maxyd=thisMax;
                    firstBC0[nc]=totTrialsBC0;
                    totTrialsBC0+=hbc2dt->GetNbinsX();
                    lastBC0[nc]=totTrialsBC0;
                }
            }else{
                mask[nc]=0;
            }
        }
    }
    TLine **vlines=new TLine*[totCases];
    TLatex **tlabels=new TLatex*[totCases+1];
    
    printf("Histos merged = %d    totTrials=%d\n",totHistos,totTrials);
    for(Int_t ja=0; ja<totCases; ja++){
        if(mask[ja]){
            printf("  %d) %s  -- %d \n",ja,histo6[ja]->GetName(),first[ja]);
            vlines[ja]=new TLine(last[ja],0.,last[ja],50000.);
            vlines[ja]->SetLineColor(kMagenta+2);
            vlines[ja]->SetLineStyle(2);
            TString ttt=histo6[ja]->GetName();
            ttt.ReplaceAll("hRawYieldTrial","");
            if(ttt.Contains("FixedMean")) ttt="Fix #mu";
            if(ttt.Contains("FixedSp20")) ttt="#sigma+";
            if(ttt.Contains("FixedSm20")) ttt="#sigma-";
            if(ttt.Contains("FreeS")) ttt="Free #sigma";
            ttt.ReplaceAll("FixedS","");
            if(bkgTreat!="" && ttt.Contains(bkgTreat.Data())) ttt.ReplaceAll(bkgTreat.Data(),"");
            tlabels[ja]=new TLatex(first[ja]+0.02*totTrials,10,ttt.Data());
            tlabels[ja]->SetTextColor(kMagenta+2);
            tlabels[ja]->SetTextColor(kMagenta+2);
        }
    }
    tlabels[totCases]=new TLatex(totTrials+30,10,"BinCnt");
    tlabels[totCases]->SetTextColor(kMagenta+2);
    tlabels[totCases]->SetTextColor(kMagenta+2);
    
    Printf("tottrials \t tottrialsBC0 \t  nBC ranges \t ");
    Printf("%d \t %d \t %d",totTrials,totTrialsBC0,nBCranges);
   
    TH1F* hRawYieldAll=new TH1F(Form("hRawYieldAll_%s", suffix),
                                " ; Trial # ; raw yield",
                                totTrials+totTrialsBC0*nBCranges,
                                0.,
                                totTrials+totTrialsBC0*nBCranges);
    TH1F* hRawYieldAllBC0=new TH1F(Form("hRawYieldAllBC0_%s", suffix),
                                   " ; Trial # ; raw yield",
                                   totTrialsBC0*nBCranges,
                                   totTrials,
                                   totTrials+totTrialsBC0*nBCranges);

    TH1F* hMeanAll6=new TH1F(Form("hMeanAll_%s", suffix),
                                   " ; Trial # ; Gaussian mean", totTrials,0., totTrials);
    TH1F* hSigmaAll6=new TH1F(Form("hSigmaAll_%s", suffix),
                              " ; Trial # ; Gaussian #sigma",totTrials,0,totTrials);
    TH1F* hChi2All6=new TH1F(Form("hChi2All_%s", suffix),
                                  " ; Trial # ; #chi^{2}",totTrials,0.,totTrials);
    hMeanAll6->SetMarkerColor(kBlue+1);
    hSigmaAll6->SetMarkerColor(kBlue+1);
    hChi2All6->SetMarkerColor(kBlue+1);
    hMeanAll6->SetLineColor(kBlue+1);
    hSigmaAll6->SetLineColor(kBlue+1);
    hChi2All6->SetLineColor(kBlue+1);
    
    Double_t lowerEdgeYieldHistos = rawYieldRef - 1.5 * rawYieldRef;
    if(lowerEdgeYieldHistos < 0) {
        lowerEdgeYieldHistos = 0;
    }
    Double_t upperEdgeYieldHistos = rawYieldRef + 1.5 * rawYieldRef;
    TH1F* hRawYieldDistAll=new TH1F("hRawYieldDistAll","  ; raw yield",200,lowerEdgeYieldHistos,upperEdgeYieldHistos);
    //hRawYieldDistAll->GetXaxis()->SetRangeUser(0.2,0.6);
    hRawYieldDistAll->SetFillStyle(3003);
    hRawYieldDistAll->SetFillColor(kBlue+1);
    TH1F* hRawYieldDistAllBC0=new TH1F("hRawYieldDistAllBC0","  ; raw yield",200,lowerEdgeYieldHistos,upperEdgeYieldHistos);
    //hRawYieldDistAllBC0->GetXaxis()->SetRangeUser(0.2,0.6);
    hRawYieldDistAllBC0->SetFillStyle(3004);
    TH1F* hStatErrDistAll=new TH1F("hStatErrDistAll","  ; Stat Unc on Yield",300,0,10000);
    TH1F* hRelStatErrDistAll=new TH1F("hRelStatErrDistAll","  ; Rel Stat Unc on Yield",100,0.,1.);
    
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
            TH1F* hmeant6=(TH1F*)fil6->Get(hmeanname.Data());
            
            TString hsigmaname=histo6[nc]->GetName();
            hsigmaname.ReplaceAll("RawYield","Sigma");
            TH1F* hsigmat6=(TH1F*)fil6->Get(hsigmaname.Data());
            
            TString hchi2name=histo6[nc]->GetName();
            hchi2name.ReplaceAll("RawYield","Chi2");
            TH1F* hchi2t6=(TH1F*)fil6->Get(hchi2name.Data());
            
            TString hbcname=histo6[nc]->GetName();
            hbcname.ReplaceAll("Trial","TrialBinC0");
            TH2F* hbc2dt060=(TH2F*)fil6->Get(hbcname.Data());
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
                    hRawYieldDistAll->Fill(ry);
                    hStatErrDistAll->Fill(ery);
                    hRelStatErrDistAll->Fill(ery/ry);
                    hRawYieldAll->SetBinContent(first[nc]+ib,ry);
                    hRawYieldAll->SetBinError(first[nc]+ib,ery);
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
                    hSigmaAll6->SetBinContent(first[nc]+ib,sig);
                    hSigmaAll6->SetBinError(first[nc]+ib,esig);
                    std::cerr << "############ MEAN " << pos << std::endl;
                    hMeanAll6->SetBinContent(first[nc]+ib,pos);
                    hMeanAll6->SetBinError(first[nc]+ib,epos);
                    hChi2All6->SetBinContent(first[nc]+ib,chi2);
                    hChi2All6->SetBinError(first[nc]+ib,0.000001);
                    if(mask[nc]==2 || mask[nc]==4){
                        for(Int_t iy=minBCrange; iy<=maxBCrange;iy++){
                            Double_t bc=hbc2dt060->GetBinContent(ib,iy);
                            Double_t ebc=hbc2dt060->GetBinError(ib,iy);
                            if(bc>0.001 && ebc<0.5*bc && bc<5.*ry){
                                Int_t theBin=iy+(firstBC0[nc]+ib-1)*nBCranges;
                                cout<< " bin content " << bc << " the bin " << theBin << " BCrange " << iy << endl;
                                hRawYieldAllBC0->SetBinContent(theBin-2,bc);
                                hRawYieldAllBC0->SetBinError(theBin-2,ebc);
                                hRawYieldDistAllBC0->Fill(bc);
                                if(hRawYieldAllBC0->GetBinCenter(theBin-2)>maxFilled) maxFilled=hRawYieldAllBC0->GetBinCenter(theBin-2);
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
    
    hRawYieldAll->SetStats(0);
    hMeanAll6->SetStats(0);
    hSigmaAll6->SetStats(0);
    hChi2All6->SetStats(0);
    hChi2All6->SetMarkerStyle(7);
    hMeanAll6->SetMinimum(0.8 * meanRef);
    hMeanAll6->SetMaximum(1.2 * meanRef);
    hSigmaAll6->SetMinimum(0.);
    //if(hSigmaAll6->GetMaximum()<0.018) hSigmaAll6->SetMaximum(0.018);
    hSigmaAll6->SetMaximum(1.1 * sigmaRef);
    hRawYieldAllBC0->SetStats(0);
    hRawYieldAllBC0->SetMarkerColor(colorBC0);
    hRawYieldAllBC0->SetLineColor(colorBC0);
    hRawYieldDistAllBC0->SetLineColor(colorBC0);
    hRawYieldDistAllBC0->SetFillColor(colorBC0);
    hRawYieldDistAllBC0->SetLineWidth(2);
    hRawYieldDistAllBC0->SetLineStyle(7);
    hRawYieldDistAllBC0->Scale(hRawYieldDistAll->GetEntries()/hRawYieldDistAllBC0->GetEntries());
    hRawYieldDistAll->SetLineWidth(2);
    
    
    TLine *l=new TLine(rawYieldRef,0.,rawYieldRef,hRawYieldDistAll->GetMaximum());
    l->SetLineColor(kRed);
    l->SetLineWidth(2);
    
    TLine *ll=new TLine(0.,rawYieldRef,totTrials+totTrialsBC0*nBCranges,rawYieldRef);
    ll->SetLineColor(kRed);
    ll->SetLineWidth(2);

    TCanvas* call=new TCanvas(Form("canvas_%s_6pad", suffix), "All",1400,800);
    call->Divide(3,2);
    call->cd(1);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    hSigmaAll6->GetYaxis()->SetTitleOffset(1.7);
    hSigmaAll6->Draw("same");
    call->cd(2);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    hMeanAll6->GetYaxis()->SetTitleOffset(1.7);
    hMeanAll6->Draw("same");
    call->cd(3);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    hChi2All6->GetYaxis()->SetTitleOffset(1.7);
    hChi2All6->Draw("same");
    call->cd(4);
    hRawYieldAll->SetTitle(title);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    Double_t newmax=1.25*(hRawYieldAll->GetMaximum()+hRawYieldAll->GetBinError(1));
    hRawYieldAll->GetYaxis()->SetTitleOffset(1.7);
    hRawYieldAll->SetMaximum(newmax);
    if(maxFilled>0) hRawYieldAll->GetXaxis()->SetRangeUser(0.,maxFilled);
    hRawYieldAll->Draw();
    hRawYieldAllBC0->Draw("same");
    ll->Draw("same");
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
    
    call->cd(5);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.06);
    hRawYieldDistAll->SetTitle(title);
    hRawYieldDistAll->Draw();
    hRawYieldDistAll->GetXaxis()->SetRangeUser(minYield*0.8,maxYield*1.2);
    hRawYieldDistAllBC0->Draw("sameshist");
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
    TLatex* tmeanBC0=new TLatex(0.15,0.79,Form("mean(BinCount)=%.3f",averBC0));
    tmeanBC0->SetNDC();
    tmeanBC0->SetTextColor(hRawYieldDistAllBC0->GetLineColor());
    tmeanBC0->Draw();
    Double_t val=hRawYieldDistAll->GetRMS();
    TLatex* thrms=new TLatex(0.15,0.62,Form("rms=%.3f  (%.2f%%)",val,val/aver*100.));
    thrms->SetNDC();
    thrms->Draw();
    val=hRawYieldDistAllBC0->GetRMS();
    TLatex* thrmsBC0=new TLatex(0.15,0.55,Form("rms(BinCount)=%.3f  (%.2f%%)",val,val/averBC0*100.));
    thrmsBC0->SetNDC();
    thrmsBC0->SetTextColor(hRawYieldDistAllBC0->GetLineColor());
    thrmsBC0->Draw();
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
    TLatex* meanRefDiffBC0=new TLatex(0.15,0.10,Form("mean(ref)-mean(BC)=%.3f  (%.2f%%)",rawYieldRef-averBC0,100.*(rawYieldRef-averBC0)/rawYieldRef));
    meanRefDiffBC0->SetNDC();
    meanRefDiffBC0->SetTextColor(hRawYieldDistAllBC0->GetLineColor());
    meanRefDiffBC0->Draw();
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
    
    
    
    
    TCanvas* c3p=new TCanvas(Form("canvas_%s_3pad", suffix), "All",1400,400);
    c3p->Divide(3,1);
    c3p->cd(1);
    gPad->SetLeftMargin(0.13);
    gPad->SetRightMargin(0.06);
    hRawYieldAll->Draw("PFC");
    hRawYieldAllBC0->Draw("PFCsame");
    ll->Draw("same");
    for(Int_t j=0; j<totCases; j++){
        if(mask[j]){
            vlines[j]->Draw("same");
            tlabels[j]->Draw();
        }
    }
    tlabels[totCases]->Draw();
    
    c3p->cd(2);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.06);
    hRawYieldDistAll->Draw();
    hRawYieldDistAllBC0->Draw("sameshist");
    l->Draw("same");
    c3p->cd(3);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.06);
    tmean->Draw();
    tmeanBC0->Draw();
    tmedian->Draw();
    thrms->Draw();
    thrmsBC0->Draw();
    tmin->Draw();
    trms->Draw();
    meanRefLabel->Draw();
    meanRefDiff->Draw();
    meanRefDiffBC0->Draw();
     //
     //
     //
     //
     //
     //
    //tup->Draw();
    //tdw->Draw();
    //tl15->Draw();
    //tl85->Draw();
    //t1s->Draw();
    c3p->SaveAs(Form("%s/3pad_pt_%s.eps", esesel.Data(), suffix));

    //   c3p->SaveAs(Form("MultiTrial_3pad.eps",iPtBin));
    //   c3p->SaveAs(Form("MultiTrial_3pad.gif",iPtBin));
    
    TCanvas* c2p=new TCanvas(Form("canvas_%s_2pad", suffix), "All",933,400);
    c2p->Divide(2,1);
    c2p->cd(1);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.06);
    hRawYieldDistAll->Draw();
    hRawYieldDistAllBC0->Draw("sameshist");
    l->Draw("same");
    c2p->cd(2);
    gPad->SetLeftMargin(0.14);
    gPad->SetRightMargin(0.06);
    tmean->Draw();
    tmeanBC0->Draw();
    tmedian->Draw();
    thrms->Draw();
    thrmsBC0->Draw();
    tmin->Draw();
    trms->Draw();
    meanRefLabel->Draw();
    meanRefDiff->Draw();
    meanRefDiffBC0->Draw();
    //
    //
    //
    //
    //
    //
    //tup->Draw();
    //tdw->Draw();
    //tl15->Draw();
    //tl85->Draw();
    //t1s->Draw();
    //   c2p->SaveAs(Form("MultiTrial_2pad.eps",iPtBin));
    c2p->SaveAs(Form("%s/2pad_pt_%s.eps", esesel.Data(), suffix));

    
    //  TString outn = filnam1.Data();
    //  outn.Prepend("Comb");
    //  cout << outn.Data() << endl;
    //  TFile *outf = new TFile(outn.Data(), "RECREATE");
    ///  call->Write();
    //  outf->Write();
    // outf->Close();
    //   TString callname = outn.Data();
    //   callname.ReplaceAll(".root",".pdf");
    //   call->Print(callname);
    //   callname.ReplaceAll(".root",".png");
    //   call->Print(callname);
    fil6->Close();
}

