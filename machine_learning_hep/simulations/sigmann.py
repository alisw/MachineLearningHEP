#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################
from array import array
from ROOT import TCanvas, TFile, gROOT, TLatex, gPad  # pylint: disable=import-error,no-name-in-module
from ROOT import TGraphErrors, TF1, TLegend # pylint: disable=import-error,no-name-in-module
import ROOT # pylint: disable=import-error,no-name-in-module

gROOT.SetBatch(True)
# pylint: disable=invalid-name

energy = [0.20, 0.90, 2.76, 5.02, 5.44, 5.50, 7.00, 8.00, 8.16, 8.80, 10.60,
          13.00, 14.00, 17., 27., 39., 63., 100.]
npoints = len(energy)
errorenergy = [0.] * npoints
sigmann = [41.6, 52.2, 61.8, 67.6, 68.4, 68.5, 70.9, 72.3, 72.5, 73.3, 75.3,
           77.6, 78.4, 80.6, 86.0, 90.5, 96.5, 102.6]
errorsigmann = [0.6, 1.0, 0.9, 0.6, 0.5, 0.5, 0.4, 0.5, 0.5, 0.6, 0.7,
                1.0, 1.1, 1.5, 2.4, 3.3, 4.6, 6.0]

energyrun5 = [7.0, 6.3, 7.0, 6.46, 5.86, 5.52]
dndetaperpairrun5 = [10., 10., 10., 10., 10., 10.]
speciesrun5 = ["O", "Ar", "Ca", "Kr", "Xe", "Pb"]
colorrun5 = [2, 4, 3, 6, 8, 19]
npartrun5 = [11.1, 24.3, 24.2, 42, 71.2, 113.7] #KRIPTON VALUE 42, IS APPROX
dndeta_points = [0., 1., 2., 3., 4., -5]
dndeta_points_min = -4
dndeta_points_max = +4


energy_ = array('f', energy)
errorenergy_ = array('f', errorenergy)
sigmann_ = array('f', sigmann)
errorsigmann_ = array('f', errorsigmann)


c1 = TCanvas("c1", "A Simple Graph with error bars", 200, 10, 700, 500)
gPad.SetLogx()
gsigma_nn = TGraphErrors(npoints, energy_, sigmann_, errorenergy_, errorsigmann_)
gsigma_nn.SetTitle(";energy(TeV);#sigma^{inel}_{NN} (mb);")
gsigma_nn.GetXaxis().SetTitleOffset(1.2)
gsigma_nn.Draw("ALP")
latex = TLatex(0.2, 0.83, "PHYSICAL REVIEW C 97, 054910 (2018), pol2 + A * ln(s^{1/2})")
latex.SetNDC()
latex.SetTextSize(0.03)
latex.Draw()
f1 = TF1("f1", "[0]+[1]*log(x)+[2]*x*x+[3]*x", 0.2, 27.)
gsigma_nn.Fit("f1", "R")
c1.SaveAs("sigmavsenergy.pdf")


for i, _ in enumerate(energy):
    print("energy(TeV)= ", energy[i], "sigma= ", sigmann[i], "+- ", errorsigmann[1])
for i, _ in enumerate(energyrun5):
    print("energy(TeV)= ", energyrun5[i], "est. sigma= %.2f" % f1.Eval(energyrun5[i]))

filedndetapbpb = TFile.Open("HEPData-ins1507090-v1-root.root")
graphpbpb05 = filedndetapbpb.Get("DN-DETARAP/Graph1D_y1")
npoint = graphpbpb05.GetN()

etaval_list = []
dndeta_list = []
errdndeta_list = []

for ip in range(npoint):
    etaval = ROOT.Double(0.)
    dndeta = ROOT.Double(0.)
    errdndeta = ROOT.Double(0.)
    graphpbpb05.GetPoint(ip, etaval, dndeta)
    errdndeta = graphpbpb05.GetErrorY(ip)
    etaval_list.append(etaval)
    dndeta_list.append(dndeta)
    errdndeta_list.append(errdndeta)

etaval_list_o = etaval_list.copy()
dndeta_list_o = dndeta_list.copy()
errdndeta_list_o = errdndeta_list.copy()

npo = len(etaval_list)
for ip in range(npo - 6, npo):
    etaval_list_o.insert(0, etaval_list[ip] * -1)
    dndeta_list_o.insert(0, dndeta_list[ip])
    errdndeta_list_o.insert(0, errdndeta_list[ip])

print(etaval_list_o)

c2 = TCanvas("c2", "A Simple Graph with error bars", 200, 10, 700, 500)
erretaval_list_d = array('f', [0.] * len(etaval_list_o))
etaval_list_d = array('f', etaval_list_o)
dndeta_list_d = array('f', dndeta_list_o)
errdndeta_list_d = array('f', errdndeta_list_o)
graphpbpb05_sym = TGraphErrors(len(etaval_list_o), etaval_list_d, dndeta_list_d, \
                                erretaval_list_d, errdndeta_list_d)
graphpbpb05_sym.SetTitle(";#eta;dN^{ch}/d#eta;")
graphpbpb05_sym.GetXaxis().SetTitleOffset(1.2)
graphpbpb05_sym.Draw("ALP")
fpbpb05 = TF1("f2", "([0]+[1]*x*x+[2]*x*x*x*x + \
               [3]/([4]*sqrt(2*3.14))*exp(-((x-[5])/(2*[4]))^2) + \
               [6]/([7]*sqrt(2*3.14))*exp(-((x-[8])/(2*[7]))^2))", -5, 5.)
fpbpb05.SetParameter(5, -1)
fpbpb05.SetParameter(4, 1)
fpbpb05.SetParameter(8, 1)
fpbpb05.SetParameter(7, 1)
fpbpb05.SetLineColor(1)
graphpbpb05_sym.Fit("f2", "R")
latex = TLatex(0.2, 0.86, "Pb-Pb 0-5% at 5.02 TeV, Phys.Lett. B 772 (2017) + fit")
latex.SetNDC()
latex.SetTextSize(0.03)
latex.Draw()
c2.SaveAs("dndetapbpb05.pdf")


f = TFile.Open("dndeta_run5.root", "recreate")
fpbpb05_norm = fpbpb05.Clone("fpbpb05_norm")
scalefactor = 1./fpbpb05_norm.Eval(0.)
fpbpb05_norm.FixParameter(0, fpbpb05_norm.GetParameter(0) * scalefactor)
fpbpb05_norm.FixParameter(1, fpbpb05_norm.GetParameter(1) * scalefactor)
fpbpb05_norm.FixParameter(2, fpbpb05_norm.GetParameter(2) * scalefactor)
fpbpb05_norm.FixParameter(3, fpbpb05_norm.GetParameter(3) * scalefactor)
fpbpb05_norm.FixParameter(6, fpbpb05_norm.GetParameter(6) * scalefactor)
for index, etap in enumerate(dndeta_points):
    print("dndeta norm at eta=%f" % etap + ", val =%.2f" % fpbpb05_norm.Eval(etap))
print("dndeta at -4<eta<4, val =%.2f" % \
    fpbpb05_norm.Integral(dndeta_points_min, dndeta_points_max))
fpbpb05_norm.Write()

listdndetafitrun5 = []
for i, species in enumerate(speciesrun5):
    listdndetafitrun5.append(fpbpb05.Clone("dndeta_fit_%s" % species))
    scalefactor = npartrun5[i] * 0.5 * dndetaperpairrun5[i]/listdndetafitrun5[i].Eval(0.)
    listdndetafitrun5[i].FixParameter(0, \
            listdndetafitrun5[i].GetParameter(0) * scalefactor)
    listdndetafitrun5[i].FixParameter(1, \
            listdndetafitrun5[i].GetParameter(1) * scalefactor)
    listdndetafitrun5[i].FixParameter(2, \
            listdndetafitrun5[i].GetParameter(2) * scalefactor)
    listdndetafitrun5[i].FixParameter(3, \
            listdndetafitrun5[i].GetParameter(3) * scalefactor)
    listdndetafitrun5[i].FixParameter(6, \
            listdndetafitrun5[i].GetParameter(6) * scalefactor)
    listdndetafitrun5[i].SetTitle(";dN^{ch}/d#eta;#eta;")
    listdndetafitrun5[i].Write()
    for index, etap in enumerate(dndeta_points):
        print("species=", species)
        print("dndeta at eta=%f" % etap + ", val =%.2f" % listdndetafitrun5[i].Eval(etap))
    print("dndeta at -4<eta<4, val =%.2f" % \
          listdndetafitrun5[i].Integral(dndeta_points_min, dndeta_points_max))

c3 = TCanvas("c3", "A Simple Graph with error bars", 200, 10, 700, 500)
gPad.SetLogy()
graphpbpb05_sym.Draw("ALP")
graphpbpb05_sym.SetMinimum(10.)
graphpbpb05_sym.SetMaximum(50000.)
leg = TLegend(.1, .7, .3, .9, "")
leg.AddEntry(graphpbpb05_sym, "XeXe 0-5% data")
for i, species in enumerate(speciesrun5):
    listdndetafitrun5[i].SetLineColor(colorrun5[i])
    leg.AddEntry(listdndetafitrun5[i], "Extrapolation %s at %.2f TeV" % \
                 (speciesrun5[i], energyrun5[i]))
    listdndetafitrun5[i].Draw("SAME")
    print("booo", listdndetafitrun5[i].Eval(0.))
leg.Draw()
c3.SaveAs("dndetapbpb_multiplespecies.pdf")
c3.SaveAs("dndetapbpb_multiplespecies.png")
