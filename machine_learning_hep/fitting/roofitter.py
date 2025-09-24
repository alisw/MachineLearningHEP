#############################################################################
##  © Copyright CERN 2024. All rights not expressly granted are reserved.  ##
##                                                                         ##
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

"""Definition of the RooFitter class and helper functions"""

from math import sqrt, isnan

import ROOT
from ROOT import RooAddPdf, RooArgList, RooArgSet, RooFit, RooRealVar, TPaveText

USE_EXTMODEL = False

# pylint: disable=too-few-public-methods, too-many-statements
# (temporary until we add more functionality)
class RooFitter:
    """Fitter using Roofit for combined fits of invariant-mass distributions"""

    def __init__(self):
        ROOT.gErrorIgnoreLevel = ROOT.kError
        #ROOT.RooMsgService.instance().setSilentMode(True)
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    def find_best_a0(self, ws, m, dh, model, old_res, range_m = None):
        kwargs = {"Save": True,
                  "PrintLevel": -1,
                  "Strategy": 2}
                  #"MaxCalls": 5000}
                  #"Minimizer": "Minuit2"}
        if range_m:
            kwargs["Range"] = (range_m[0], range_m[1])
        tmp_frame = m.frame()
        dh.plotOn(tmp_frame, ROOT.RooFit.Name("data"))
        model.plotOn(tmp_frame)
        chi2 = tmp_frame.chiSquare()
        a0_var = ws.var("a0")  # Get the a0 parameter
        attempt = 0
        a0_values = [10, 20, 30, 40, 50, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500,
                     700, 1000, 1500, 2000, 2500, 3000, 5000, 10000]

        res = old_res
        chi_threshold = 6.
        while (chi2 > chi_threshold or isnan(chi2)) and attempt < len(a0_values):
            print(f"Attempt {attempt+1}: Setting a0 to {a0_values[attempt]}")
            a0_var.setVal(a0_values[attempt])  # Change a0 value
            attempt += 1

            res = model.fitTo(dh, **kwargs)
            tmp_frame = m.frame()
            dh.plotOn(tmp_frame)
            model.plotOn(tmp_frame)
            chi2 = tmp_frame.chiSquare()

        if chi2 <= chi_threshold:
            print(f"Fit improved: chi2 = {chi2}, stopping adjustments.")

        return res, model

    # pylint: disable=too-many-branches
    def fit_mass_new(
        self, hist, pdfnames: dict, param_names: dict, fit_spec: dict, level: str,
        fixed_sigma: bool = False, fixed_sigma_val: float = 0., roows: ROOT.RooWorkspace = None, plot: bool = False
    ):
        """New fit method"""
        print(f"fit spec:\n{fit_spec}")
        print(f'fitting for level {level} pt: {fit_spec.get("ptrange", "")}')
        if hist.GetEntries() == 0:
            raise UserWarning("Cannot fit histogram with no entries")
        ws = roows or ROOT.RooWorkspace("ws")
        var_m = fit_spec.get("var", "m")

        hist_integral = hist.Integral(*(hist.FindBin(mmass) for mmass in fit_spec.get("range")))
        if "data" in level:
            print(f"hist integral: {hist_integral}")
        #n_signal = RooRealVar("n_signal", "Number of signal events", 0.3 * hist_integral, 0., 1.2 * hist_integral)
        #n_background = RooRealVar("n_background", "Number of background events",
                                   #0.3 * hist_integral, 0., 1.2 * hist_integral)
        n_signal = RooRealVar("n_signal", "Number of signal events", 1000, 100, 100000000)
        n_background = RooRealVar("n_background", "Number of background events", 1000, 100, 100000000)

        model = None
        for comp, spec in fit_spec.get("components", {}).items():
            fn = ws.factory(spec["fn"])
            if comp == "model":
                model = fn
        if model is None:
            raise ValueError("model not set")

        m = ws.var(var_m)
        print(f"m var: {m}")

        if level == "mc":
            sigma_sgn = ws.var(param_names["gauss_sigma"])
            if fixed_sigma:
                sigma_sgn.setVal(fixed_sigma_val)
                sigma_sgn.setConstant(True)

        if level == "data":
            signal_pdf = ws.pdf(pdfnames["pdf_sig"])
            if not signal_pdf:
                raise ValueError("sig PDF not found")
            background_pdf = ws.pdf(pdfnames["pdf_bkg"])
            if not background_pdf:
                raise ValueError("bkg pdf not found")
            model = RooAddPdf(
                "model", "Total model", RooArgList(signal_pdf, background_pdf), RooArgList(n_signal, n_background)
            )

        dh = ROOT.RooDataHist("dh", "dh", [m], Import=hist)
        if range_m := fit_spec.get("range"):
            m.setRange("fit", *range_m)
            print(f'using fit range: {range_m}, var range: {m.getRange("fit")}')
            res = model.fitTo(dh, Range=(range_m[0], range_m[1]), Save=True, PrintLevel=-1, Strategy=2)
            ret_model = model
            #if level == "data" and USE_EXTMODEL:
            #    for v in ws.allVars():
            #        v.setConstant(True)
            #    res, ret_model = self.find_best_a0(ws, m, dh, extmodel, res, range_m)
            #elif
            if level == "data":
                res, ret_model = self.find_best_a0(ws, m, dh, model, res, range_m)
        else:
            res = model.fitTo(dh, Save=True, PrintLevel=-1, Strategy=2)
            ret_model = model
            #if level == "data" and USE_EXTMODEL:
            #    for v in ws.allVars():
            #        v.setConstant(True)
            #    res, ret_model = self.find_best_a0(ws, m, dh, extmodel, res)
            #elif
            if level == "data":
                res, ret_model = self.find_best_a0(ws, m, dh, model, res)
        frame = None
        residual_frame = None
        if plot:
            c = ROOT.TCanvas()
            c.SetLogy()
            c.cd()
            frame = m.frame()
            dh.plotOn(frame, ROOT.RooFit.Name("data"))
            ret_model.plotOn(frame)
            ret_model.paramOn(frame, Layout=(0.65, 1.0, 0.9))
            frame.getAttText().SetTextFont(42)
            frame.getAttText().SetTextSize(0.001)
            if range_m:
                frame.SetAxisRange(range_m[0], range_m[1], "X")
            frame.SetAxisRange(0.0, frame.GetMaximum() + (frame.GetMaximum() * 0.3), "Y")

            try:
                for pdf in ret_model.pdfList():
                    pdf_name = pdf.GetName()
                    ret_model.plotOn(
                        frame,
                        ROOT.RooFit.Components(pdf),
                        ROOT.RooFit.Name(f"pdf_{pdf_name}"),
                        ROOT.RooFit.LineStyle(ROOT.ELineStyle.kDashed),
                        ROOT.RooFit.LineColor(ROOT.kViolet),
                        ROOT.RooFit.LineWidth(1),
                    )
                    # model.SetName("bkg")
                ret_model.plotOn(frame, ROOT.RooFit.Name("model"))
            except:  # pylint: disable=bare-except  # noqa: E722
                pass
            # for comp in fit_spec.get('components', {}):
            #     if comp != 'model':
            #         model.plotOn(frame, ROOT.RooFit.Components(comp),
            #                      ROOT.RooFit.LineStyle(ROOT.ELineStyle.kDashed))
            # c.Modified()
            # c.Update()

        if level == "data": #and USE_EXTMODEL and frame is not None:
            residuals = frame.residHist("data", "pdf_bkg")
            residual_frame = m.frame()
            residual_frame.addPlotable(residuals, "P")

            n_signal_ext = ROOT.RooRealVar("n_signal_ext", "Expected signal events", n_signal.getVal(), 0, 1e6)
            signal_pdf_ext = ROOT.RooExtendPdf("signal_pdf_ext", "Extended signal PDF", signal_pdf, n_signal_ext)

            signal_pdf_ext.plotOn(
                residual_frame,
                ROOT.RooFit.LineColor(ROOT.kBlue),
                ROOT.RooFit.Normalization(1.0, ROOT.RooAbsReal.RelativeExpected),
            )

            if range_m:
                residual_frame.SetAxisRange(range_m[0], range_m[1], "X")
            residual_frame.SetYTitle("Residuals")

        return (res, ws, frame, residual_frame, dh, ret_model)

    def fit_mass(self, hist, fit_spec, plot=False):
        """Old fit method"""
        if hist.GetEntries() == 0:
            raise UserWarning("Cannot fit histogram with no entries")
        ws = ROOT.RooWorkspace("ws")

        model = None
        for comp, spec in fit_spec.get("components", {}).items():
            ws.factory(spec["fn"])
            if comp == "sum":
                model = ws.pdf(comp)
        if model is None:
            raise ValueError("model not set")

        m = ws.var("m")
        # m.setRange('full', 0., 3.)
        dh = ROOT.RooDataHist("dh", "dh", [m], Import=hist)
        # model = ws.pdf('sum')
        # model.Print('t')
        res = model.fitTo(dh, Save=True, PrintLevel=-1)
        frame = m.frame() if plot else None
        if plot:
            dh.plotOn(frame)  # , ROOT.RooFit.Range(0., 3.))
            model.plotOn(frame)
            model.paramOn(frame)
            for comp in fit_spec.get("components", {}):
                if comp != "sum":
                    model.plotOn(frame, ROOT.RooFit.Components(comp), ROOT.RooFit.LineStyle(ROOT.ELineStyle.kDashed))
        return (res, ws, frame)


def estimate_signal(hist, fit_spec, n_bkg, bkg_integral):
    bin_min = hist.FindBin(fit_spec["mass_mean"] - 3 * fit_spec["sigma_signal"])
    bin_max = hist.FindBin(fit_spec["mass_mean"] + 3 * fit_spec["sigma_signal"])
    msum = 0.
    for ind in range(bin_min, bin_max + 1):
        msum += hist.GetBinContent(ind)
    bkg = calculate_background(n_bkg, bkg_integral)
    return msum - bkg


def calculate_background(n_bkg, bkg_integral):
    return n_bkg.getVal() * bkg_integral

def calc_signif(roows, res, pdfnames, param_names, mean_sgn, sigma_sgn):
    """Calculate significance, signal, background, signal/background ratio."""
    #if not USE_EXTMODEL:
    #    return (0., 0., 0., 0., 0., 0, 0, 0.)
    f_sig = roows.pdf(pdfnames["pdf_sig"])
    n_signal = res.floatParsFinal().find("n_signal").getVal()
    sigma_n_signal = res.floatParsFinal().find("n_signal").getError()

    # Code to subtract reflections from the final significance
    # frac_refl = roows.var("frac_refl")
    # n_signal = res.floatParsFinal().find("n_signal").getVal()*(1-frac_refl.getVal())
    # sigma_n_signal = res.floatParsFinal().find("n_signal").getError()*(1-frac_refl.getVal())

    f_bkg = roows.pdf(pdfnames["pdf_bkg"])
    n_bkg = res.floatParsFinal().find("n_background").getVal()
    sigma_n_bkg = res.floatParsFinal().find("n_background").getError()

    massvar = roows.var(param_names["mass"])
    massvar.setRange("signal", mean_sgn.getVal() - 3 * sigma_sgn.getVal(), mean_sgn.getVal() + 3 * sigma_sgn.getVal())

    massvar_set = RooArgSet(massvar)
    norm_set = RooFit.NormSet(massvar_set)
    signal_range = RooFit.Range("signal")
    signal_integral = f_sig.createIntegral(massvar_set, norm_set, signal_range)
    bkg_integral = f_bkg.createIntegral(massvar_set, norm_set, signal_range)

    n_signal_signal = signal_integral.getVal() * n_signal
    n_bkg_signal = bkg_integral.getVal() * n_bkg

    print(f"n signal signal: {n_signal_signal} n bkg signal: {n_bkg_signal}")

    if n_signal_signal + n_bkg_signal == 0.:
        significance = 0.0
    else:
        significance = n_signal_signal / sqrt(n_signal_signal + n_bkg_signal)

    # Calculate the error on the signal and bkg integrals using the covariance matrix
    sigma_signal_integral = signal_integral.getPropagatedError(res)
    sigma_bkg_integral = bkg_integral.getPropagatedError(res)

    sigma_n_signal_signal = sqrt(
        (signal_integral.getVal() * sigma_n_signal) ** 2 + (n_signal * sigma_signal_integral) ** 2
    )
    sigma_n_bkg_signal = sqrt((bkg_integral.getVal() * sigma_n_bkg) ** 2 + (n_bkg * sigma_bkg_integral) ** 2)

    if n_signal_signal + n_bkg_signal == 0.:
        dS_dS = 0.0
        dS_dB = 0.0
    else:
        dS_dS = (1 / sqrt(n_signal_signal + n_bkg_signal) -
                 (n_signal_signal / (2 * (n_signal_signal + n_bkg_signal)**(3/2))))
        dS_dB = -n_signal_signal / (2 * (n_signal_signal + n_bkg_signal)**(3/2))
    significance_err = sqrt(
            (dS_dS * sigma_n_signal_signal) ** 2 +
            (dS_dB * sigma_n_bkg_signal) ** 2)

    #Signal to bkg ratio
    if n_bkg_signal == 0.:
        s_over_b = 0.0
        s_over_b_err = 0.0 # as S/B is ill-defined
    elif n_signal_signal == 0.:
        s_over_b = 0.0
        s_over_b_err = s_over_b * sqrt((sigma_n_bkg_signal / n_bkg_signal) ** 2)
    else:
        s_over_b = n_signal_signal / n_bkg_signal
        s_over_b_err = (
                s_over_b * sqrt((sigma_n_signal_signal / n_signal_signal) ** 2 +
                                (sigma_n_bkg_signal / n_bkg_signal) ** 2))

    return (
        n_signal_signal,
        sigma_n_signal_signal,
        n_bkg_signal,
        sigma_n_bkg_signal,
        significance,
        significance_err,
        s_over_b,
        s_over_b_err,
    )


def create_text_info(x_1, y_1, x_2, y_2):
    """Create an info box for fit plots and set its style."""
    text_info = TPaveText(x_1, y_1, x_2, y_2, "NDC")
    text_info.SetBorderSize(0)
    text_info.SetFillColor(0)  # Transparent fill
    text_info.SetFillStyle(0)
    text_info.SetTextAlign(12)
    text_info.SetTextFont(42)  # Helvetica
    text_info.SetTextSize(0.035)
    text_info.SetTextColor(4)

    return text_info


def add_text_info_fit(text_info, frame, roows, param_names):
    """Add fit info on the info box."""
    chi2 = frame.chiSquare()
    mean_sgn = roows.var(param_names["gauss_mean"])
    sigma_sgn = roows.var(param_names["gauss_sigma"])
    sigmawide_sgn = roows.var(param_names["double_gauss_sigma"])
    refl_frac = roows.var(param_names["fraction_refl"])
    text_info.AddText(f"#chi^{{2}}/ndf = {chi2:.2f}")
    text_info.AddText(f"#mu = {mean_sgn.getVal():.3f} #pm {mean_sgn.getError():.3f}")
    text_info.AddText(f"#sigma = {sigma_sgn.getVal():.4f} #pm {sigma_sgn.getError():.4f}")
    if sigmawide_sgn:
        text_info.AddText(f"#sigma wide = {sigmawide_sgn.getVal():.4f} #pm {sigmawide_sgn.getError():.4f}")
    if refl_frac:
        text_info.AddText(f"refl.frac. = {refl_frac.getVal():.4f} #pm {refl_frac.getError():.4f}")
    if a0 := roows.var("a0"):
        text_info.AddText(f"a0 = {a0.getVal():.3f} #pm {a0.getError():.3f}")
    if a1 := roows.var("a1"):
        text_info.AddText(f"a1 = {a1.getVal():.3f} #pm {a1.getError():.3f}")
    if a2 := roows.var("a2"):
        text_info.AddText(f"a2 = {a2.getVal():.3f} #pm {a2.getError():.3f}")
    if offset := roows.var("offset"):
        text_info.AddText(f"offset = {offset.getVal():.3f} #pm {offset.getError():.3f}")


def add_text_info_perf(text_info, sig, sig_err, bkg, bkg_err, s_over_b, s_over_b_err, signif, signif_err):
    """Add signal, background, signal/background and significance on the info box."""
    text_info.AddText(f"S(3#sigma) = {sig:.0f} #pm {sig_err:.0f}")
    text_info.AddText(f"B(3#sigma) = {bkg:.0f} #pm {bkg_err:.0f}")
    text_info.AddText(f"S/B(3#sigma) = {s_over_b:.3f} #pm {s_over_b_err:.3f}")
    text_info.AddText(f"Signif(3#sigma) = {signif:.1f} #pm {signif_err:.1f}")
