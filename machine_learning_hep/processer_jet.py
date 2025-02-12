#  © Copyright CERN 2024. All rights not expressly granted are reserved.  #
#                                                                         #
# This program is free software: you can redistribute it and/or modify it #
#  under the terms of the GNU General Public License as published by the  #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version. This program is distributed in the hope that #
#  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  #
#     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#           See the GNU General Public License for more details.          #
#    You should have received a copy of the GNU General Public License    #
#   along with this program. if not, see <https://www.gnu.org/licenses/>. #

import copy
import functools
import itertools
import math

import numpy as np
import pandas as pd
from ROOT import TH1F, TFile

from machine_learning_hep.processer import Processer
from machine_learning_hep.utilities import dfquery, read_df
from machine_learning_hep.utils.hist import bin_array, create_hist, fill_hist, get_axis, get_range, project_hist


# pylint: disable=too-many-instance-attributes, too-many-statements
class ProcesserJets(Processer):
    species = "processer"

    def __init__(
        self,
        case,
        datap,
        run_param,
        mcordata,
        p_maxfiles,  # pylint: disable=too-many-arguments
        d_root,
        d_pkl,
        d_pklsk,
        d_pkl_ml,
        p_period,
        i_period,
        p_chunksizeunp,
        p_chunksizeskim,
        p_maxprocess,
        p_frac_merge,
        p_rd_merge,
        d_pkl_dec,
        d_pkl_decmerged,
        d_results,
        typean,
        runlisttrigger,
        d_mcreweights,
    ):
        super().__init__(
            case,
            datap,
            run_param,
            mcordata,
            p_maxfiles,
            d_root,
            d_pkl,
            d_pklsk,
            d_pkl_ml,
            p_period,
            i_period,
            p_chunksizeunp,
            p_chunksizeskim,
            p_maxprocess,
            p_frac_merge,
            p_rd_merge,
            d_pkl_dec,
            d_pkl_decmerged,
            d_results,
            typean,
            runlisttrigger,
            d_mcreweights,
        )
        self.logger.info("initialized processer for HF jets")

        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]

        # bins: 2d array [[low, high], ...]
        self.bins_skimming = np.array(list(zip(self.lpt_anbinmin, self.lpt_anbinmax)), "d")  # TODO: replace with cfg
        self.bins_analysis = np.array(list(zip(self.lpt_finbinmin, self.lpt_finbinmax)), "d")

        # skimming bins in overlap with the analysis range
        self.active_bins_skim = [
            iskim
            for iskim, ptrange in enumerate(self.bins_skimming)
            if ptrange[0] < max(self.bins_analysis[:, 1]) and ptrange[1] > min(self.bins_analysis[:, 0])
        ]
        self.logger.info("Using skimming bins: %s", self.active_bins_skim)

        # binarray: array of bin edges as double (passable to ROOT)
        limits_mass = datap["analysis"][self.typean]["mass_fit_lim"]
        binwidth_mass = datap["analysis"][self.typean]["bin_width"]
        nbins_mass = int(round((limits_mass[1] - limits_mass[0]) / binwidth_mass))
        self.binarray_mass = bin_array(nbins_mass, limits_mass[0], limits_mass[1])
        self.binarray_ptjet = np.asarray(self.cfg("bins_ptjet"), "d")
        self.binarray_pthf = np.asarray(self.cfg("sel_an_binmin", []) + self.cfg("sel_an_binmax", [])[-1:], "d")
        self.binarrays_obs = {"gen": {}, "det": {}}
        self.binarrays_ptjet = {"gen": {}, "det": {}}
        for obs in self.cfg("observables", {}):
            var = obs.split("-")
            for v in var:
                if v in self.binarrays_obs:
                    continue
                for level in ("gen", "det"):
                    if binning := self.cfg(f"observables.{v}.bins_{level}_var"):
                        self.binarrays_obs[level][v] = np.asarray(binning, "d")
                    elif binning := self.cfg(f"observables.{v}.bins_{level}_fix"):
                        self.binarrays_obs[level][v] = bin_array(*binning)
                    elif binning := self.cfg(f"observables.{v}.bins_var"):
                        self.binarrays_obs[level][v] = np.asarray(binning, "d")
                    elif binning := self.cfg(f"observables.{v}.bins_fix"):
                        self.binarrays_obs[level][v] = bin_array(*binning)
                    else:
                        self.logger.error("no binning specified for %s, using defaults", v)
                        self.binarrays_obs[level][v] = bin_array(10, 0.0, 1.0)

                    if binning := self.cfg(f"observables.{v}.bins_ptjet"):
                        self.binarrays_ptjet[level][v] = np.asarray(binning, "d")
                    else:
                        self.binarrays_ptjet[level][v] = self.binarray_ptjet
        self.binarrays_obs["gen"]["fPt"] = self.binarray_pthf
        self.binarrays_obs["det"]["fPt"] = self.binarray_pthf
        self.binarrays_ptjet["gen"]["fPt"] = np.asarray(self.cfg("bins_ptjet_eff"), "d")
        self.binarrays_ptjet["det"]["fPt"] = np.asarray(self.cfg("bins_ptjet_eff"), "d")

    # region observables
    # pylint: disable=invalid-name
    def _verify_variables(self, dfi):
        """
        Explicit (slow) implementation, use for reference/validation only
        """
        df = dfi.copy(deep=True)
        df["rg"] = -0.1
        df["nsd"] = -1.0
        df["zg"] = -0.1
        for idx, row in df.iterrows():
            isSoftDropped = False
            nsd = 0
            for zg, theta in zip(row["zg_array"], row["fTheta"]):
                if zg >= self.cfg("zcut", 0.1):
                    if not isSoftDropped:
                        df.loc[idx, "zg"] = zg
                        df.loc[idx, "rg"] = theta
                        isSoftDropped = True
                    nsd += 1
            df.loc[idx, "nsd"] = nsd
        for var in ["zg", "nsd", "rg"]:
            if np.allclose(dfi[var], df[var]):
                self.logger.info("%s check ok", var)
            else:
                self.logger.error("%s check failed", var)
                mask = np.isclose(dfi[var], df[var])
                print(df[~mask][var], flush=True)
                print(dfi[~mask][var], flush=True)

    def _calculate_variables(self, df, verify=False):  # pylint: disable=invalid-name
        self.logger.info("calculating variables")
        if len(df) == 0:
            df["nsub21"] = None
            df["zg"] = None
            df["rg"] = None
            df["nsd"] = None
            df["lnkt"] = None
            df["lntheta"] = None
            return df
        df["nsub21"] = df.fNSub2 / df.fNSub1
        # TODO: catch nsub1 == 0
        self.logger.debug("zg")
        df["zg_array"] = np.array(0.5 - abs(df.fPtSubLeading / (df.fPtLeading + df.fPtSubLeading) - 0.5))
        zcut = self.cfg("zcut", 0.1)
        df["zg"] = df["zg_array"].apply(lambda ar: next((zg for zg in ar if zg >= zcut), -0.1))
        df["rg"] = df[["zg_array", "fTheta"]].apply(
            (lambda ar: next((rg for (zg, rg) in zip(ar.zg_array, ar.fTheta) if zg >= zcut), -0.1)), axis=1
        )
        df["nsd"] = df["zg_array"].apply(lambda ar: len([zg for zg in ar if zg >= zcut]))

        self.logger.debug("Lund")
        df["lnkt"] = df[["fPtSubLeading", "fTheta"]].apply(
            (lambda ar: np.log(ar.fPtSubLeading * np.sin(ar.fTheta))), axis=1
        )
        df["lntheta"] = df["fTheta"].apply(lambda x: -np.log(x))
        # df['lntheta'] = np.array(-np.log(df.fTheta))

        self.logger.info("EEC")
        df["eecweight"] = df[["fPairPt", "fJetPt"]].apply((lambda ar: ar.fPairPt / ar.fJetPt**2), axis=1)

        if self.cfg("hfjet", True):
            df["dr"] = np.sqrt(
                (df.fJetEta - df.fEta) ** 2 + ((df.fJetPhi - df.fPhi + math.pi) % math.tau - math.pi) ** 2
            )
            df["jetPx"] = df.fJetPt * np.cos(df.fJetPhi)
            df["jetPy"] = df.fJetPt * np.sin(df.fJetPhi)
            df["jetPz"] = df.fJetPt * np.sinh(df.fJetEta)
            df["hfPx"] = df.fPt * np.cos(df.fPhi)
            df["hfPy"] = df.fPt * np.sin(df.fPhi)
            df["hfPz"] = df.fPt * np.sinh(df.fEta)
            df["zpar_num"] = df.jetPx * df.hfPx + df.jetPy * df.hfPy + df.jetPz * df.hfPz
            df["zpar_den"] = df.jetPx * df.jetPx + df.jetPy * df.jetPy + df.jetPz * df.jetPz
            df["zpar"] = df.zpar_num / df.zpar_den
            df[df["zpar"] >= 1.0]["zpar"] = 0.999  # move 1 to last bin

        self.logger.debug("done")
        if verify:
            self._verify_variables(df)
        return df

    def split_df(self, dfi, frac):
        """split data frame based on df number"""
        # dfa = dfi.split(frac=frac, random_state=1234)
        # return dfa, dfi.drop(dfa.index)
        mask = (dfi.index.get_level_values(0) % 100) < frac * 100
        return dfi[mask], dfi[~mask]

    # region histomass
    # pylint: disable=too-many-branches
    def process_histomass_single(self, index):
        self.logger.info("Processing (histomass) %s", self.l_evtorig[index])

        with TFile.Open(self.l_histomass[index], "recreate") as _:
            dfevtorig = read_df(self.l_evtorig[index])
            histonorm = TH1F("histonorm", "histonorm", 4, 0, 4)
            histonorm.SetBinContent(1, len(dfquery(dfevtorig, self.s_evtsel)))
            if self.l_collcnt:
                dfcollcnt = read_df(self.l_collcnt[index])
                ser_collcnt = dfcollcnt[self.cfg(f"counter_read_{self.mcordata}")]
                collcnt_read = functools.reduce(lambda x, y: float(x) + float(y), (ar[0] for ar in ser_collcnt))
                self.logger.info("sampled %g collisions", collcnt_read)
                histonorm.SetBinContent(2, collcnt_read)
                ser_collcnt = dfcollcnt[self.cfg("counter_tvx")]
                collcnt_tvx = functools.reduce(lambda x, y: float(x) + float(y), (ar[0] for ar in ser_collcnt))
                histonorm.SetBinContent(3, collcnt_tvx)
            if self.l_bccnt:
                dfbccnt = read_df(self.l_bccnt[index])
                ser_bccnt = dfbccnt[self.cfg("counter_tvx")]
                bccnt_tvx = functools.reduce(lambda x, y: float(x) + float(y), (ar[0] for ar in ser_bccnt))
                histonorm.SetBinContent(4, bccnt_tvx)
            get_axis(histonorm, 0).SetBinLabel(1, "N_{evt}")
            get_axis(histonorm, 0).SetBinLabel(2, "N_{coll}")
            get_axis(histonorm, 0).SetBinLabel(3, "N_{coll}^{TVX}")
            get_axis(histonorm, 0).SetBinLabel(4, "N_{BC}^{TVX}")
            histonorm.Write()

            df = pd.concat(read_df(self.mptfiles_recosk[bin][index]) for bin in self.active_bins_skim)
            # remove entries outside of kinematic range (should be taken care of by projections in analyzer)
            df = df.loc[(df.fJetPt >= min(self.binarray_ptjet)) & (df.fJetPt < max(self.binarray_ptjet))]
            df = df.loc[(df.fPt >= min(self.bins_analysis[:, 0])) & (df.fPt < max(self.bins_analysis[:, 1]))]

            # Custom skimming cuts
            df = self.apply_cuts_all_ptbins(df)

            if col_evtidx := self.cfg("cand_collidx"):
                h = create_hist("h_ncand", ";N_{cand}", 20, 0.0, 20.0)
                fill_hist(h, df.groupby([col_evtidx]).size(), write=True)

            h = create_hist(
                "h_mass-ptjet-pthf",
                ";M (GeV/#it{c}^{2});p_{T}^{jet} (GeV/#it{c});p_{T}^{HF} (GeV/#it{c})",
                self.binarray_mass,
                self.binarray_ptjet,
                self.binarray_pthf,
            )
            fill_hist(h, df[["fM", "fJetPt", "fPt"]], write=True)

            for sel_name, sel_spec in self.cfg("data_selections", {}).items():
                if sel_spec["level"] == self.mcordata:
                    df_sel = dfquery(df, sel_spec["query"])
                    h = create_hist(
                        f"h_mass-ptjet-pthf_{sel_name}",
                        ";M (GeV/#it{c}^{2});p_{T}^{jet} (GeV/#it{c});p_{T}^{HF} (GeV/#it{c})",
                        self.binarray_mass,
                        self.binarray_ptjet,
                        self.binarray_pthf,
                    )
                    fill_hist(h, df_sel[["fM", "fJetPt", "fPt"]], write=True)

            if self.mcordata == "mc":
                df, _ = self.split_df(df, self.cfg("frac_mcana", 0.2))
                if len(df) == 0:
                    return
                self.logger.debug("MC det: %s", df.index.get_level_values(0).unique())
                if f := self.cfg("closure.exclude_feeddown_det"):
                    dfquery(df, f, inplace=True)
                if f := self.cfg("closure.filter_reflections"):
                    dfquery(df, f, inplace=True)
                if self.cfg("closure.use_matched"):
                    if idx := self.cfg("efficiency.index_match"):
                        df["idx_match"] = df[idx].apply(lambda ar: ar[0] if len(ar) > 0 else -1)
                        dfquery(df, "idx_match >= 0", inplace=True)

            self._calculate_variables(df)

            for obs, spec in self.cfg("observables", {}).items():
                self.logger.info("preparing histograms for %s", obs)
                var = obs.split("-")
                if not all(v in df for v in var):
                    self.logger.error("dataframe does not contain %s", var)
                    continue
                h = create_hist(
                    f"h_mass-ptjet-pthf-{obs}",
                    f";M (GeV/#it{{c}}^{{2}});p_{{T}}^{{jet}} (GeV/#it{{c}});p_{{T}}^{{HF}} (GeV/#it{{c}});{obs}",
                    self.binarray_mass,
                    self.binarray_ptjet,
                    self.binarray_pthf,
                    *[self.binarrays_obs["det"][v] for v in var],
                )
                for i, v in enumerate(var):
                    get_axis(h, 3 + i).SetTitle(self.cfg(f"observables.{v}.label", v))

                fill_hist(h, df[["fM", "fJetPt", "fPt", *var]], arraycols=spec.get("arraycols", None), write=True)

    # TODO:
    # - binning variations (separate ranges for MC and data)
    # - priors (reweight response matrix)

    # region efficiency
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    def process_efficiency_single(self, index):
        self.logger.info("Processing (efficiency) %s", self.l_evtorig[index])

        cats = ["pr", "np"]
        levels_eff = ["gen", "det", "genmatch", "detmatch", "detmatch_gencuts"]
        levels_effkine = ["gen", "det"]
        cuts = ["nocuts", "cut"]
        observables = self.cfg("observables", {})
        observables.update({"fPt": {"label": "p_{T}^{HF} (GeV/#it{c})"}})
        h_eff = {
            (cat, level): create_hist(
                f"h_ptjet-pthf_{cat}_{level}",
                ";p_{T}^{jet} (GeV/#it{c});p_{T}^{HF} (GeV/#it{c})",
                self.binarrays_ptjet["det"]["fPt"],
                self.binarray_pthf,
            )
            for cat in cats
            for level in levels_eff
        }
        h_response = {}
        h_effkine = {}
        h_response_fd = {}
        h_effkine_fd = {}
        h_mctruth = {}
        for cat in cats:
            for obs in self.cfg("observables", {}):
                self.logger.info("preparing response matrix for %s", obs)
                var = obs.split("-")
                dim = len(var) + 1
                h_response[(cat, obs)] = h = create_hist(
                    f"h_response_{cat}_{obs}",
                    f"response matrix {obs}",
                    self.binarrays_ptjet["det"][var[0]],
                    *[self.binarrays_obs["det"][v] for v in var],
                    self.binarrays_ptjet["gen"][var[0]],
                    *[self.binarrays_obs["gen"][v] for v in var],
                    self.binarray_pthf,
                )
                get_axis(h, 0).SetTitle("p_{T}^{jet} (GeV/#it{c})")
                get_axis(h, dim).SetTitle("p_{T}^{jet} (GeV/#it{c})")
                get_axis(h, 2 * dim).SetTitle("p_{T}^{HF} (GeV/#it{c})")
                for i, v in enumerate(var, 1):
                    get_axis(h, i).SetTitle(self.cfg(f"observables.{v}.label", v))
                    get_axis(h, i + dim).SetTitle(self.cfg(f"observables.{v}.label", v))
                for cut in cuts:
                    h_effkine[(cat, "det", cut, obs)] = he = project_hist(h, list(range(dim)), {}).Clone()
                    he.SetName(f"h_effkine_{cat}_det_{cut}_{obs}")
                    h_effkine[(cat, "gen", cut, obs)] = he = project_hist(h, list(range(dim, 2 * dim)), {}).Clone()
                    he.SetName(f"h_effkine_{cat}_gen_{cut}_{obs}")
                h_mctruth[(cat, obs)] = create_hist(
                    f"h_ptjet-pthf-{obs}_{cat}_gen",
                    f";p_{{T}}^{{jet}} (GeV/#it{{c}});p_{{T}}^{{HF}} (GeV/#it{{c}});{obs}",
                    self.binarrays_ptjet["gen"][var[0]],
                    self.binarray_pthf,
                    *[self.binarrays_obs["gen"][v] for v in var],
                )
                h_response_fd[obs] = create_hist(
                    f"h_response_fd_{obs}",
                    f";response matrix fd {obs}",
                    self.binarrays_ptjet["det"][var[0]],
                    self.binarrays_obs["det"]["fPt"],
                    *[self.binarrays_obs["det"][v] for v in var],
                    self.binarrays_ptjet["gen"][var[0]],
                    self.binarrays_obs["gen"]["fPt"],
                    *[self.binarrays_obs["gen"][v] for v in var],
                )
                for level, cut in itertools.product(levels_effkine, cuts):
                    h_effkine_fd[(level, cut, obs)] = create_hist(
                        f"h_effkine_fd_{level}_{cut}_{obs}",
                        f"effkine {obs}",
                        self.binarrays_ptjet[level][var[0]],
                        self.binarrays_obs[level]["fPt"],
                        *[self.binarrays_obs[level][v] for v in var],
                    )

        # create partial versions for closure testing
        h_effkine_frac = copy.deepcopy(h_effkine)
        h_response_frac = copy.deepcopy(h_response)
        for hist in itertools.chain(h_effkine_frac.values(), h_response_frac.values()):
            hist.SetName(hist.GetName() + "_frac")

        with TFile.Open(self.l_histoeff[index], "recreate") as rfile:
            # TODO: avoid hard-coding values here (check if restriction is needed at all)
            cols = (
                None
                if not self.cfg("hfjet", True)
                else [
                    "ismcprompt",
                    "ismcsignal",
                    "ismcfd",
                    "fPt",
                    "fEta",
                    "fPhi",
                    "fJetPt",
                    "fJetEta",
                    "fJetPhi",
                    "fPtLeading",
                    "fPtSubLeading",
                    "fTheta",
                    "fNSub2DR",
                    "fNSub1",
                    "fNSub2",
                    "fJetNConstituents",
                    "fEnergyMother",
                    "fPairTheta",
                    "fPairPt",
                ]
            )

            # read generator level
            dfgen_orig = pd.concat(
                read_df(self.mptfiles_gensk[bin][index], columns=cols) for bin in self.active_bins_skim
            )
            df = self._calculate_variables(dfgen_orig)
            df = df.rename(lambda name: name + "_gen", axis=1)
            if self.cfg("hfjet", True):
                dfgen = {
                    "pr": df.loc[(df.ismcsignal_gen == 1) & (df.ismcprompt_gen == 1)],
                    "np": df.loc[(df.ismcsignal_gen == 1) & (df.ismcfd_gen == 1)],
                }
            else:
                dfgen = {"pr": df, "np": df}

            # read detector level
            if cols:
                cols.extend(self.cfg("efficiency.extra_cols", []))
                if idx := self.cfg("efficiency.index_match"):
                    cols.append(idx)
            df = pd.concat(read_df(self.mptfiles_recosk[bin][index], columns=cols) for bin in self.active_bins_skim)

            # Custom skimming cuts
            df = self.apply_cuts_all_ptbins(df)

            dfquery(df, self.cfg("efficiency.filter_det"), inplace=True)
            if idx := self.cfg("efficiency.index_match"):
                df["idx_match"] = df[idx].apply(lambda ar: ar[0] if len(ar) > 0 else -1)
            else:
                self.logger.warning("No matching criterion specified, cannot match det and gen")
            df = self._calculate_variables(df)
            if self.cfg("hfjet", True):
                dfdet = {
                    "pr": df.loc[(df.ismcsignal == 1) & (df.ismcprompt == 1)],
                    "np": df.loc[(df.ismcsignal == 1) & (df.ismcfd == 1)],
                }
            else:
                dfdet = {"pr": df, "np": df}

            dfmatch = {
                cat: pd.merge(dfdet[cat], dfgen[cat], left_on=["df", "idx_match"], right_index=True)
                for cat in cats
                if "idx_match" in dfdet[cat]
            }

            for cat in cats:
                fill_hist(h_eff[(cat, "gen")], dfgen[cat][["fJetPt_gen", "fPt_gen"]])
                fill_hist(h_eff[(cat, "det")], dfdet[cat][["fJetPt", "fPt"]])
                if cat in dfmatch and dfmatch[cat] is not None:
                    df = dfmatch[cat]
                    fill_hist(h_eff[(cat, "genmatch")], df[["fJetPt_gen", "fPt_gen"]])
                    fill_hist(h_eff[(cat, "detmatch")], df[["fJetPt", "fPt"]])
                    # apply gen-level cuts for Run 2 efficiencies
                    range_ptjet_gen = get_range(h_eff[(cat, "gen")], 0)
                    range_pthf_gen = get_range(h_eff[(cat, "gen")], 1)
                    df = df.loc[(df.fJetPt_gen >= range_ptjet_gen[0]) & (df.fJetPt_gen < range_ptjet_gen[1])]
                    df = df.loc[(df.fPt_gen >= range_pthf_gen[0]) & (df.fPt_gen < range_pthf_gen[1])]
                    fill_hist(h_eff[(cat, "detmatch_gencuts")], df[["fJetPt", "fPt"]])
                else:
                    self.logger.error("No matching, could not fill matched detector-level histograms")

            for obs, cat in itertools.product(observables, cats):
                if cat in dfmatch and dfmatch[cat] is not None:
                    self._prepare_response(dfmatch[cat], h_effkine, h_response, cat, obs)
                    f = self.cfg("frac_mcana", 0.2)
                    _, df_mccorr = self.split_df(dfmatch[cat], f if f < 1.0 else 0.0)
                    self._prepare_response(df_mccorr, h_effkine_frac, h_response_frac, cat, obs)
                    self._prepare_response_fd(dfmatch[cat], h_effkine_fd, h_response_fd, obs)

                # TODO: move outside of loop?
                if self.cfg("closure.use_matched"):
                    self.logger.info("using matched for truth")
                    df_mcana, _ = self.split_df(dfmatch[cat], self.cfg("frac_mcana", 0.2))
                else:
                    df_mcana, _ = self.split_df(dfgen[cat], self.cfg("frac_mcana", 0.2))
                if f := self.cfg("closure.exclude_feeddown_gen"):
                    self.logger.debug("excluding feeddown gen")
                    dfquery(df_mcana, f, inplace=True)

                arraycols = [i - 3 for i in self.cfg(f"observables.{obs}.arraycols", [])]
                var = obs.split("-")
                self.logger.debug(
                    "Observable %s has arraycols %s -> %s", obs, arraycols, [var[icol] for icol in arraycols]
                )
                df_mcana = self._explode_arraycols(df_mcana, [var[icol] for icol in arraycols])
                fill_hist(h_mctruth[(cat, obs)], df_mcana[["fJetPt_gen", "fPt_gen", *(f"{v}_gen" for v in var)]])

            for name, obj in itertools.chain(
                h_eff.items(),
                h_effkine.items(),
                h_response.items(),
                h_effkine_fd.items(),
                h_response_fd.items(),
                h_effkine_frac.items(),
                h_response_frac.items(),
                h_mctruth.items(),
            ):
                try:
                    rfile.WriteObject(obj, obj.GetName())
                except Exception as ex:  # pylint: disable=broad-exception-caught
                    self.logger.error("Writing of <%s> (%s) failed: %s", name, str(obj), str(ex))

    def _explode_arraycols(self, df: pd.DataFrame, arraycols: "list[str]") -> pd.DataFrame:
        if len(arraycols) > 0:
            self.logger.debug("Exploding columns %s", arraycols)
            # only consider rows with corresponding det- and gen-level entries
            df["length"] = [len(x) for x in df[arraycols[0]]]
            df["length_gen"] = [len(x) for x in df[arraycols[0] + "_gen"]]
            df = df.loc[df.length == df.length_gen]
            df = df.explode(arraycols + [col + "_gen" for col in arraycols])
            df.dropna(inplace=True)
        return df

    def _prepare_response(self, dfi, h_effkine, h_response, cat, obs):
        var = obs.split("-")
        dim = len(var) + 1
        axes_det = [get_axis(h_response[(cat, obs)], i) for i in range(dim)]
        axes_gen = [get_axis(h_response[(cat, obs)], i) for i in range(dim, 2 * dim)]
        arraycols = [i - 3 for i in self.cfg(f"observables.{obs}", {}).get("arraycols", [])]

        df = dfi
        df = self._explode_arraycols(df, [var[icol] for icol in arraycols])

        df = df.loc[(df.fJetPt >= axes_det[0].GetXmin()) & (df.fJetPt < axes_det[0].GetXmax())]
        for i, v in enumerate(var, 1):
            df = df.loc[(df[v] >= axes_det[i].GetXmin()) & (df[v] < axes_det[i].GetXmax())]
        fill_hist(h_effkine[(cat, "det", "nocuts", obs)], df[["fJetPt", *var]])
        df = df.loc[(df.fJetPt >= axes_gen[0].GetXmin()) & (df.fJetPt < axes_gen[0].GetXmax())]
        for i, v in enumerate(var, 1):
            df = df.loc[(df[f"{v}_gen"] >= axes_gen[i].GetXmin()) & (df[f"{v}_gen"] < axes_gen[i].GetXmax())]
        fill_hist(h_effkine[(cat, "det", "cut", obs)], df[["fJetPt", *var]])

        # print(df[['fJetPt', *var, 'fJetPt_gen', *(f'{v}_gen' for v in var), 'fPt']].info(), flush=True)
        fill_hist(h_response[(cat, obs)], df[["fJetPt", *var, "fJetPt_gen", *(f"{v}_gen" for v in var), "fPt"]])

        df = dfi
        df = self._explode_arraycols(df, [var[icol] for icol in arraycols])
        df = df.loc[(df.fJetPt >= axes_gen[0].GetXmin()) & (df.fJetPt < axes_gen[0].GetXmax())]
        for i, v in enumerate(var, 1):
            df = df.loc[(df[f"{v}_gen"] >= axes_gen[i].GetXmin()) & (df[f"{v}_gen"] < axes_gen[i].GetXmax())]
        fill_hist(h_effkine[(cat, "gen", "nocuts", obs)], df[["fJetPt_gen", *(f"{v}_gen" for v in var)]])
        df = df.loc[(df.fJetPt >= axes_det[0].GetXmin()) & (df.fJetPt < axes_det[0].GetXmax())]
        for i, v in enumerate(var, 1):
            df = df.loc[(df[v] >= axes_det[i].GetXmin()) & (df[v] < axes_det[i].GetXmax())]
        fill_hist(h_effkine[(cat, "gen", "cut", obs)], df[["fJetPt_gen", *(f"{v}_gen" for v in var)]])

    def _prepare_response_fd(self, dfi, h_effkine, h_response, obs):
        var = obs.split("-")
        dim = len(var) + 2
        axes_det = [get_axis(h_response[obs], i) for i in range(dim)]
        axes_gen = [get_axis(h_response[obs], i) for i in range(dim, 2 * dim)]
        arraycols = [i - 3 for i in self.cfg(f"observables.{obs}", {}).get("arraycols", [])]

        df = dfi
        df = self._explode_arraycols(df, [var[icol] for icol in arraycols])
        # TODO: the first cut should be taken care of by under-/overflow bins, check their usage in analyzer
        df = df.loc[
            (df.fJetPt >= axes_det[0].GetXmin())
            & (df.fJetPt < axes_det[0].GetXmax())
            & (df.fPt >= axes_det[1].GetXmin())
            & (df.fPt < axes_det[1].GetXmax())
        ]
        for i, v in enumerate(var, 2):
            df = df.loc[(df[v] >= axes_det[i].GetXmin()) & (df[v] < axes_det[i].GetXmax())]
        fill_hist(h_effkine[("det", "nocuts", obs)], df[["fJetPt", "fPt", *var]])
        df = df.loc[
            (df.fJetPt_gen >= axes_gen[0].GetXmin())
            & (df.fJetPt_gen < axes_gen[0].GetXmax())
            & (df.fPt_gen >= axes_gen[1].GetXmin())
            & (df.fPt_gen < axes_gen[1].GetXmax())
        ]
        for i, v in enumerate(var, 2):
            df = df.loc[(df[f"{v}_gen"] >= axes_gen[i].GetXmin()) & (df[f"{v}_gen"] < axes_gen[i].GetXmax())]
        fill_hist(h_effkine[("det", "cut", obs)], df[["fJetPt", "fPt", *var]])

        fill_hist(h_response[obs], df[["fJetPt", "fPt", *var, "fJetPt_gen", "fPt_gen", *(f"{v}_gen" for v in var)]])

        df = dfi
        df = self._explode_arraycols(df, [var[icol] for icol in arraycols])
        df = df.loc[
            (df.fJetPt_gen >= axes_gen[0].GetXmin())
            & (df.fJetPt_gen < axes_gen[0].GetXmax())
            & (df.fPt_gen >= axes_gen[1].GetXmin())
            & (df.fPt_gen < axes_gen[1].GetXmax())
        ]
        for i, v in enumerate(var, 2):
            df = df.loc[(df[f"{v}_gen"] >= axes_gen[i].GetXmin()) & (df[f"{v}_gen"] < axes_gen[i].GetXmax())]
        fill_hist(h_effkine[("gen", "nocuts", obs)], df[["fJetPt_gen", "fPt", *(f"{v}_gen" for v in var)]])
        df = df.loc[
            (df.fJetPt >= axes_det[0].GetXmin())
            & (df.fJetPt < axes_det[0].GetXmax())
            & (df.fPt >= axes_det[1].GetXmin())
            & (df.fPt < axes_det[1].GetXmax())
        ]
        for i, v in enumerate(var, 2):
            df = df.loc[(df[v] >= axes_det[i].GetXmin()) & (df[v] < axes_det[i].GetXmax())]
        fill_hist(h_effkine[("gen", "cut", obs)], df[["fJetPt_gen", "fPt", *(f"{v}_gen" for v in var)]])
