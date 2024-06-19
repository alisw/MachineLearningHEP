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
    def fit_mass(self, hist, fit_spec, plot = False):
        if hist.GetEntries() == 0:
            raise UserWarning('Cannot fit histogram with no entries')
        ws = ROOT.RooWorkspace("ws")
        for comp, spec in fit_spec.get('components', {}).items():
            ws.factory(spec['fn'])
        m = ws.var('m')
        dh = ROOT.RooDataHist("dh", "dh", [m], Import=hist)
        model = ws.pdf('sum')
        model.fitTo(dh, PrintLevel=-1)
        frame = m.frame() if plot else None
        if plot:
            dh.plotOn(frame)
            model.plotOn(frame)
            for comp in fit_spec.get('components', {}):
                if comp != 'sum':
                    model.plotOn(frame, ROOT.RooFit.Components(comp),
                                 ROOT.RooFit.LineStyle(ROOT.ELineStyle.kDashed))
        return (ws, frame)
