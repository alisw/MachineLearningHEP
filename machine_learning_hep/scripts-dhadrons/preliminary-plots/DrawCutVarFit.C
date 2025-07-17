#include "TCanvas.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraphAsymmErrors.h"
#include "TH1.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TPad.h"
#include "TStyle.h"
#include <iostream>

using namespace std;

void SetStyle();
void SetStyleHisto(TH1D *h);
void SetStyleHisto(TH1F *h);
void NormaliseHist1d(TH1 *h);

//const Int_t colors[] = {kGreen + 2, kBlue - 4, kRed, kOrange + 7};
//const Int_t markers[] = {20, 21, 33, 34};
//const Int_t npoints[] = {5, 3, 4, 4, 4, 4, 4};
//const Int_t nPtBins = 11;
//const Double_t ptlimsmiddle[11] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
//                                   7.5, 9,   11,  14,  20};
//const Int_t nPtBinsCoarse = 11;
//Double_t ptlimsCoarse[nPtBinsCoarse + 1] = {1., 2., 3.,  4.,  5.,  6.,
//                                            7., 8., 10., 12., 16., 24.};
//Double_t ptbinwidthCoarse[nPtBinsCoarse] = {1., 1., 1., 1., 1., 1.,
//                                            1., 2., 2., 4., 8.};
//const Double_t ptlimsmiddlePrompt[21] = {
//    0.5,  1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
//    6.25, 6.75, 7.25, 7.75, 8.5,  9.5,  11.,  14.,  20.,  30.};
//Double_t yvaluncPrompt[21] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
//                              0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

std::vector<Double_t> bdtScoreCuts_1_2 = {0.21, 0.24, 0.27, 0.30, 0.33, 0.35, 0.37, 0.39, 0.41, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.58};
std::vector<Double_t> bdtScoreCuts_2_3 = {0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52};
std::vector<Double_t> bdtScoreCuts_3_4 = {0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60};
std::vector<Double_t> bdtScoreCuts_4_5 = {0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.38, 0.40, 0.43, 0.47, 0.50, 0.54, 0.58};
std::vector<Double_t> bdtScoreCuts_5_6 = {0.10, 0.12, 0.14, 0.16, 0.18, 0.21, 0.24, 0.26, 0.28, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.50, 0.52};
std::vector<Double_t> bdtScoreCuts_6_8 = {0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.36, 0.39, 0.41, 0.43, 0.46, 0.49, 0.52};
std::vector<Double_t> bdtScoreCuts_8_12 = {0.08, 0.11, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.41, 0.43, 0.46, 0.49};
//std::vector<Double_t> bdtScoreCuts = {0.29, 0.33, 0.37, 0.41, 0.45, 0.49,
//                                      0.53, 0.57, 0.61, 0.65, 0.69, 0.73,
//                                      0.77, 0.81, 0.85, 0.89, 0.93};
std::vector<Double_t> bdtScoreCuts_toPlot = {0.29, 0.45, 0.61, 0.77, 0.93};
std::vector<Double_t> bdtScoreCuts_toPlot_ind = {0, 4, 8, 12, 16};

const Int_t binMin = 4;
const Int_t binMax = 5;
std::vector<Double_t> bdtScoreCuts = bdtScoreCuts_4_5;

bool DrawAllPoints = false;

void DrawCutVarFit(bool isPreliminary = kTRUE) {

  //TGaxis::SetMaxDigits(1);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);

  TFile *CutVarFile = nullptr;

  // D
  TH1F *hRawYieldsVsCutPt = nullptr;
  TH1F *hRawYieldPromptVsCut = nullptr;
  TH1F *hRawYieldFDVsCut = nullptr;
  TH1F *hRawYieldsVsCutReSum = nullptr;

  CutVarFile =
      new TFile("/data8/majak/systematics/230824/CutVarLc_pp13TeV_LHC24d3_default.root",
                  "read");
  hRawYieldsVsCutPt =
      (TH1F *)CutVarFile->Get(Form("hRawYieldVsCut_pt%d_%d", binMin, binMax));
  hRawYieldPromptVsCut =
      (TH1F *)CutVarFile->Get(Form("hRawYieldPromptVsCut_pt%d_%d", binMin, binMax));
  hRawYieldFDVsCut =
      (TH1F *)CutVarFile->Get(Form("hRawYieldNonPromptVsCut_pt%d_%d", binMin, binMax));
  hRawYieldsVsCutReSum =
      (TH1F *)CutVarFile->Get(Form("hRawYieldSumVsCut_pt%d_%d", binMin, binMax));

  SetStyleHisto(hRawYieldsVsCutPt);
  SetStyleHisto(hRawYieldPromptVsCut);
  SetStyleHisto(hRawYieldFDVsCut);
  SetStyleHisto(hRawYieldsVsCutReSum);

  hRawYieldsVsCutPt->SetMarkerStyle(20);
  hRawYieldsVsCutPt->SetMarkerSize(1);
  hRawYieldsVsCutPt->SetMarkerColor(kBlack);
  hRawYieldsVsCutPt->SetLineColor(kBlack);

  hRawYieldPromptVsCut->SetMarkerStyle(33);
  hRawYieldPromptVsCut->SetMarkerSize(1);
  hRawYieldPromptVsCut->SetMarkerColor(kRed + 1);
  hRawYieldPromptVsCut->SetLineColor(kRed + 1);

  hRawYieldFDVsCut->SetMarkerStyle(33);
  hRawYieldFDVsCut->SetMarkerSize(1);
  hRawYieldFDVsCut->SetMarkerColor(kAzure + 4);
  hRawYieldFDVsCut->SetLineColor(kAzure + 4);

  hRawYieldsVsCutReSum->SetMarkerStyle(33);
  hRawYieldsVsCutReSum->SetMarkerSize(1);
  hRawYieldsVsCutReSum->SetMarkerColor(kGreen + 2);
  hRawYieldsVsCutReSum->SetLineColor(kGreen + 2);

  hRawYieldsVsCutPt->GetYaxis()->SetTitle("Raw yield");
  hRawYieldsVsCutPt->GetYaxis()->SetTitleSize(0.05);
  hRawYieldsVsCutPt->GetYaxis()->SetMaxDigits(3);
  hRawYieldsVsCutPt->GetXaxis()->SetTitle("Minimum BDT score for non-prompt#Lambda_{c}^{#plus}");
  hRawYieldsVsCutPt->GetXaxis()->SetTitleSize(0.05);
  hRawYieldsVsCutPt->SetMinimum(0.1);
  hRawYieldsVsCutPt->SetMaximum(35000);
  hRawYieldsVsCutPt->SetLineWidth(2);
  hRawYieldsVsCutPt->GetYaxis()->SetTitleOffset(1.1);
  // Set custom labels
    for (size_t i = 0; i < bdtScoreCuts.size(); ++i) {
      hRawYieldsVsCutPt->GetXaxis()->SetBinLabel(i + 1, Form(""));
      for (size_t j = 0; j < bdtScoreCuts_toPlot_ind.size(); ++j)
        //if (bdtScoreCuts[i] == bdtScoreCuts_toPlot[j]) {
        if (i == bdtScoreCuts_toPlot_ind[j]) {
          std::cout << "bdtScoreCuts[i] " << bdtScoreCuts[i] << " bdtScoreCuts_toPlot " << bdtScoreCuts_toPlot_ind[j] << std::endl;
          hRawYieldsVsCutPt->GetXaxis()->SetBinLabel(i + 1, Form("%.2f",bdtScoreCuts[i]));
        }
    }

  TCanvas *c1 = new TCanvas("c1", "c1", 0, 0, 750, 750);
  gStyle->SetOptStat(0);
  c1->SetTickx();
  c1->SetTicky();
  c1->SetBottomMargin(0.13);
  c1->SetLeftMargin(0.17);
  c1->SetTopMargin(0.06);
  c1->SetRightMargin(0.06);
  c1->cd();

  hRawYieldsVsCutPt->Draw();
  hRawYieldPromptVsCut->Draw("HISTsame");
  hRawYieldPromptVsCut->SetFillStyle(3154);
  hRawYieldPromptVsCut->SetFillColor(kRed + 1);
  hRawYieldFDVsCut->Draw("HISTsame");
  hRawYieldFDVsCut->SetFillStyle(3145);
  hRawYieldFDVsCut->SetFillColor(kAzure + 4);
  hRawYieldsVsCutReSum->Draw("HISTsame");

  TLatex info;
  info.SetNDC();
  info.SetTextFont(43);
  info.SetTextSize(40);
  info.DrawLatex(0.21, 0.86, "ALICE Preliminary");

  TLatex infos;
  infos.SetNDC();
  infos.SetTextFont(43);
  infos.SetTextSize(30);
  infos.DrawLatex(0.21, 0.80,
                  "#Lambda_{c}^{#plus} and charge conj., pp, #sqrt{#it{s}} = 13.6 TeV");
  //infos.DrawLatex(0.21, 0.74, "|#it{y}| < 0.5");

  TLatex infoPt;
  infoPt.SetNDC();
  infoPt.SetTextFont(43);
  infoPt.SetTextSize(30);

  infoPt.DrawLatex(0.62, 0.70, Form("%d < #it{p}_{T} < %d GeV/#it{c}", binMin, binMax));
  //  TLatex info5;
  //  info5.SetNDC();
  //  info5.SetTextFont(43);
  //  info5.SetTextSize(15);
  //  info5.DrawLatex(0.48, 0.66, "#it{f} (b #rightarrow B^{0}, b #rightarrow
  //  B^{+})_{LHCb}, BR (H_{b} #rightarrow D^{0}+X)_{PYTHIA 8}");//,
  //  info1.DrawLatex(0.5, 0.74-0.02, "average of");
  // info.DrawLatex(0.20, 0.70, "#Lambda_{c}^{+} #rightarrow pK^{0}_{S}");
  //  if (isPreliminary){
  // info.DrawLatex(0.28, 0.85, "ALICE");
  // info.DrawLatex(0.28, 0.85, "ALICE");
  // info.DrawLatex(0.22, 0.2-0.06, "Preliminary");
  //    }

  //  TLatex info2;
  //  info2.SetNDC();
  //  info2.SetTextFont(43);
  //  info2.SetTextSize(15);
  //  info2.DrawLatex(0.21, 0.17, "#pm 3.7% lumi. unc. not shown");
  //  info2.DrawLatex(0.21, 0.22, "#pm 0.76% BR unc. not shown");

  TLegend *leg = new TLegend(0.62, 0.48, 0.70, 0.68);
  leg->SetFillColor(0);
  leg->SetFillStyle(0);
  leg->SetBorderSize(0);
  leg->SetMargin(0.46);
  leg->SetTextSize(28);
  leg->SetTextFont(43);
  leg->AddEntry(hRawYieldsVsCutPt, "Data", "p");
  leg->AddEntry(hRawYieldPromptVsCut, "Prompt", "F");
  leg->AddEntry(hRawYieldFDVsCut, "Non-prompt", "F");
  leg->AddEntry(hRawYieldsVsCutReSum, "Total", "l");
  leg->Draw();

  c1->SaveAs(Form("./CutVarFitLcFD_%d-%d.pdf", binMin, binMax));
  c1->SaveAs(Form("./CutVarFitLcFD_%d-%d.png", binMin, binMax));
  c1->SaveAs(Form("./CutVarFitLcFD_%d-%d.eps", binMin, binMax));
}

void SetStyle() {
  cout << "Setting style!" << endl;

  gStyle->Reset("Plain");
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);
  gStyle->SetCanvasColor(10);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetFrameLineWidth(1);
  gStyle->SetFrameFillColor(kWhite);
  gStyle->SetPadColor(10);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.13);
  gStyle->SetPadTopMargin(0.07);
  gStyle->SetPadRightMargin(0.07);
  gStyle->SetHistLineWidth(1);
  gStyle->SetHistLineColor(kRed);
  gStyle->SetFuncWidth(2);
  gStyle->SetFuncColor(kGreen);
  gStyle->SetLineWidth(2);
  gStyle->SetLabelSize(0.055, "xyz");
  gStyle->SetLabelOffset(0.01, "y");
  gStyle->SetLabelOffset(0.01, "x");
  gStyle->SetLabelColor(kBlack, "xyz");
  // gStyle->SetTitleSize(0.055,"xyz");
  // gStyle->SetTitleOffset(1.5,"y");
  // gStyle->SetTitleOffset(1.15,"x");
  gStyle->SetTitleFillColor(kWhite);
  gStyle->SetTextSizePixels(30);
  gStyle->SetTextFont(42);
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendFillColor(kWhite);
  gStyle->SetLegendFont(42);
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(0.7);
  gStyle->SetMarkerColor(kBlack);
}

void SetStyleHisto(TH1D *h) {

  h->SetLineColor(kBlack);
  h->SetLineWidth(2);
  h->GetYaxis()->SetLabelFont(42);
  h->GetYaxis()->SetTitleFont(42);
  h->GetYaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetTitleOffset(1.7);
  h->GetYaxis()->SetLabelSize(0.05);
  h->GetYaxis()->SetDecimals(kTRUE);
  // h->GetYaxis()->SetNdivisions(507);
  h->GetXaxis()->SetTitleFont(42);
  h->GetXaxis()->SetLabelFont(42);
  h->GetXaxis()->SetTitleSize(0.06);
  h->GetXaxis()->SetTitleOffset(1.2);
  h->GetXaxis()->SetLabelSize(0.07);
  h->GetXaxis()->SetNdivisions(510);
}

void SetStyleHisto(TH1F *h) {

  h->SetLineColor(kBlack);
  h->SetLineWidth(2);
  h->GetYaxis()->SetLabelFont(42);
  h->GetYaxis()->SetTitleFont(42);
  h->GetYaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetTitleOffset(1.7);
  h->GetYaxis()->SetLabelSize(0.05);
  h->GetYaxis()->SetDecimals(kTRUE);
  // h->GetYaxis()->SetNdivisions(507);
  h->GetXaxis()->SetTitleFont(42);
  h->GetXaxis()->SetLabelFont(42);
  h->GetXaxis()->SetTitleSize(0.06);
  h->GetXaxis()->SetTitleOffset(1.3);
  h->GetXaxis()->SetLabelSize(0.07);
  h->GetXaxis()->SetLabelOffset(0.01);
  //  h->GetXaxis()->SetNdivisions(505);
  // h->GetXaxis()->SetNdivisions(510);
}

void NormaliseHist1d(TH1 *h) {
  if (h) {
    // dN/dpt
    for (Int_t i = 1; i <= h->GetNbinsX(); i++) {
      h->SetBinContent(i,
                       h->GetBinContent(i) / (h->GetXaxis()->GetBinWidth(i)));
      //		hnew->SetBinError(i,hnew->GetBinContent(i)/(hnew->GetBinWidth(i)
      //* TMath::Sqrt(hnew->GetBinContent(i)))); // may need to look at again
      h->SetBinError(i, h->GetBinError(i) / (h->GetXaxis()->GetBinWidth(i)));
    }
  } else {
    cout << "can't normalise hist - not found" << endl;
  }
}
