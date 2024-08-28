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

import ROOT

# pylint: disable=too-few-public-methods
# (temporary until we add more functionality)
class RooFitter:
    def __init__(self):
        ROOT.RooMsgService.instance().setSilentMode(True)
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    def fit_mass_new(self, hist, fit_spec, roows = None, plot = False):
        if hist.GetEntries() == 0:
            raise UserWarning('Cannot fit histogram with no entries')
        ws = roows or ROOT.RooWorkspace("ws")
        var_m = fit_spec.get('var', 'm')
        for comp, spec in fit_spec.get('components', {}).items():
            fn = ws.factory(spec['fn'])
            if comp == 'model':
                model = fn
        m = ws.var(var_m)
        # if range_m := fit_spec.get('range'):
        #     m.setRange(range_m[0], range_m[1])
        dh = ROOT.RooDataHist("dh", "dh", [m], Import=hist)
        if range_m := fit_spec.get('range'):
            m.setRange('fit', *range_m)
            # print(f'using fit range: {range_m}, var range: {m.getRange("fit")}')
            res = model.fitTo(dh, Range=(range_m[0], range_m[1]), Save=True, PrintLevel=-1)
            # model.Print('v')
        else:
            res = model.fitTo(dh, Save=True, PrintLevel=-1)
        frame = None
        if plot:
            c = ROOT.TCanvas()
            c.SetLogy()
            c.cd()
            frame = m.frame()
            dh.plotOn(frame)
            model.plotOn(frame)
            model.paramOn(frame, Layout=(.65,1.,.9))
            frame.getAttText().SetTextFont(42)
            frame.getAttText().SetTextSize(.03)
            try:
                for pdf in model.pdfList():
                    model.plotOn(frame, ROOT.RooFit.Components(pdf),
                                 ROOT.RooFit.LineStyle(ROOT.ELineStyle.kDashed),
                                 ROOT.RooFit.LineColor(ROOT.kViolet),
                                 ROOT.RooFit.LineWidth(1))
            # pylint: disable=bare-except
            except:
                pass
            # for comp in fit_spec.get('components', {}):
            #     if comp != 'model':
            #         model.plotOn(frame, ROOT.RooFit.Components(comp),
            #                      ROOT.RooFit.LineStyle(ROOT.ELineStyle.kDashed))
            # c.Modified()
            # c.Update()
        return (res, ws, frame)


    def fit_mass(self, hist, fit_spec, plot = False):
        if hist.GetEntries() == 0:
            raise UserWarning('Cannot fit histogram with no entries')
        ws = ROOT.RooWorkspace("ws")
        for comp, spec in fit_spec.get('components', {}).items():
            ws.factory(spec['fn'])
            if comp == 'sum':
                model = ws.pdf(comp)
        m = ws.var('m')
        # m.setRange('full', 0., 3.)
        dh = ROOT.RooDataHist("dh", "dh", [m], Import=hist)
        # model = ws.pdf('sum')
        # model.Print('t')
        res = model.fitTo(dh, Save=True, PrintLevel=-1)
        frame = m.frame() if plot else None
        if plot:
            dh.plotOn(frame) #, ROOT.RooFit.Range(0., 3.))
            model.plotOn(frame)
            model.paramOn(frame)
            for comp in fit_spec.get('components', {}):
                if comp != 'sum':
                    model.plotOn(frame, ROOT.RooFit.Components(comp),
                                 ROOT.RooFit.LineStyle(ROOT.ELineStyle.kDashed))
        return (res, ws, frame)
