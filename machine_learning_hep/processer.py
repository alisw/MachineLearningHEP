#############################################################################
##  © Copyright CERN 2024. All rights not expressly granted are reserved.  ##
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

"""
main script for doing data processing, machine learning and analysis
"""
import glob
import multiprocessing as mp
import os
import pickle
import random as rd
import re
import sys
import tempfile
from copy import deepcopy
from functools import reduce
from pandas.api.types import is_numeric_dtype

import numpy as np
import pandas as pd
import uproot

from .bitwise import tag_bit_df
from .io import dump_yaml_from_dict
from .logger import get_logger
from .utilities import (count_df_length_pkl, dfquery, mask_df, merge_method,
                        mergerootfiles, openfile, read_df, seldf_singlevar,
                        write_df)
from .utilities_files import (appendmainfoldertolist, create_folder_struc,
                              createlist, list_folders)

pd.options.mode.chained_assignment = None

class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'
    logger = get_logger()

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments, consider-using-f-string
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles, # pylint: disable=too-many-branches
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, typean, runlisttrigger, d_mcreweights):
        self.doml = datap["doml"]
        self.case = case
        self.typean = typean
        #directories
        self.d_prefix_ml = datap["ml"].get("prefix_dir_ml", "")
        self.d_root = d_root
        self.d_pkl = d_pkl
        self.d_pklsk = d_pklsk
        self.d_pkl_ml = d_pkl_ml
        self.d_results = d_results
        self.d_mcreweights = d_mcreweights
        self.datap = datap
        self.mcordata = mcordata

        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(self.lpt_anbinmin)

        self.p_frac_merge = p_frac_merge
        try:
            iter(p_frac_merge)
        except TypeError:
            self.p_frac_merge = [p_frac_merge] * self.p_nptbins
        if len(self.p_frac_merge) != self.p_nptbins:
            print(f"Length of merge-fraction list != number of pT bins \n" \
                    f"{len(self.p_frac_merge)} != {self.p_nptbins}")
            sys.exit(1)

        self.p_rd_merge = p_rd_merge
        self.period = p_period
        self.i_period = i_period
        self.select_period = datap["multi"][mcordata]["select_period"]
        self.select_jobs = datap["multi"][mcordata].get("select_jobs", None)
        if self.select_jobs:
            self.select_jobs = [f"{job}/" for job in self.select_jobs[i_period]]

        self.run_param = run_param
        self.p_maxfiles = p_maxfiles
        self.p_chunksizeunp = p_chunksizeunp
        self.p_chunksizeskim = p_chunksizeskim

        self.df_read = datap['dfs']['read']
        self.df_merge = datap['dfs'].get('merge', None)
        self.df_write = datap['dfs'].get('write', None)

        #parameter names
        self.p_maxprocess = p_maxprocess
        self.indexsample = None
        self.p_dofullevtmerge = datap["dofullevtmerge"]
        #namefile root
        self.n_root = datap["files_names"]["namefile_unmerged_tree"]

        #namefiles pkl
        # def nget(d : dict, k : list, dd = None):
        #     return nget(d.get(k.pop(0), {}), k, dd) if len(k) > 1 else d.get(k.pop(0), dd)
        # nget(datap, ['dfs', 'write', 'jetsubdet', 'file'])
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_collcnt = datap["files_names"]["namefile_collcnt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_evt_count_ml = datap["files_names"].get("namefile_evt_count", "evtcount.yaml")
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.n_mcreweights = datap["files_names"]["namefile_mcweights"]

        #selections
        self.s_reco_skim = datap["sel_reco_skim"]
        self.s_gen_skim = datap["sel_gen_skim"]

        #bitmap
        self.b_mcrefl = datap["bitmap_sel"].get("ismcrefl", None)

        #variables name
        self.v_train = datap["variables"]["var_training"]
        self.v_bitvar = datap["bitmap_sel"]["var_name"]
        self.v_bitvar_gen = datap["bitmap_sel"]["var_name_gen"]
        self.v_bitvar_origgen = datap["bitmap_sel"]["var_name_origgen"]
        self.v_bitvar_origrec = datap["bitmap_sel"]["var_name_origrec"]
        self.v_candtype = datap["var_cand"]
        self.v_swap = datap.get("var_swap", None)
        self.v_isstd = datap["bitmap_sel"]["var_isstd"]
        self.v_ismcsignal = datap["bitmap_sel"]["var_ismcsignal"]
        self.v_ismcprompt = datap["bitmap_sel"]["var_ismcprompt"]
        self.v_ismcfd = datap["bitmap_sel"]["var_ismcfd"]
        self.v_ismcbkg = datap["bitmap_sel"]["var_ismcbkg"]
        self.v_ismcrefl = datap["bitmap_sel"]["var_ismcrefl"]
        self.v_var_binning = datap["var_binning"]
        self.v_invmass = datap["variables"].get("var_inv_mass", "inv_mass")
        self.v_rapy = datap["variables"].get("var_y", "y_cand")

        #list of files names
        if os.path.isdir(self.d_root):
            self.l_path = list_folders(self.d_root, self.n_root, self.p_maxfiles,
                                       self.select_jobs)
        elif glob.glob(f"{self.d_pkl}/**/{self.n_reco}", recursive=True):
            self.l_path = list_folders(self.d_pkl, self.n_reco, self.p_maxfiles,
                                       self.select_jobs)
        else:
            self.n_sk = self.n_reco.replace(".p", "_%s%d_%d.p" % \
                          (self.v_var_binning, self.lpt_anbinmin[0], self.lpt_anbinmax[0]))
            self.l_path = list_folders(self.d_pklsk, self.n_sk, self.p_maxfiles,
                                       self.select_jobs)

        self.l_root = createlist(self.d_root, self.l_path, self.n_root)
        self.l_reco = createlist(self.d_pkl, self.l_path, self.n_reco)
        self.l_evt = createlist(self.d_pkl, self.l_path, self.n_evt)
        self.l_evtorig = createlist(self.d_pkl, self.l_path, self.n_evtorig)
        self.l_collcnt = createlist(self.d_pkl, self.l_path, self.n_collcnt)
        self.l_histomass = createlist(self.d_results, self.l_path, self.n_filemass)
        self.l_histoeff = createlist(self.d_results, self.l_path, self.n_fileeff)
        self.l_historesp = createlist(self.d_results, self.l_path, self.n_fileresp)

        if self.mcordata == "mc":
            self.l_gen = createlist(self.d_pkl, self.l_path, self.n_gen)

        self.f_totevt = os.path.join(self.d_pkl, self.n_evt)
        self.f_totevtorig = os.path.join(self.d_pkl, self.n_evtorig)

        self.p_modelname = datap["mlapplication"]["modelname"]
        # Analysis pT bins
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.lpt_model = datap["mlapplication"]["modelsperptbin"]
        self.dirmodel = self.d_prefix_ml + datap["ml"]["mlout"]
        self.mltype = datap["ml"]["mltype"]
        self.class_labels = datap["ml"].get("class_labels", None)
        self.lpt_model = appendmainfoldertolist(self.dirmodel, self.lpt_model)
        # Potentially mask certain values (e.g. nsigma TOF of -999)
        self.p_mask_values = datap["ml"].get("mask_values", None)

        self.lpt_probcutpre = datap["mlapplication"]["probcutpresel"][self.mcordata]
        self.lpt_probcutfin = datap["analysis"][self.typean].get("probcuts", None)

        self.bins_skimming = np.array(list(zip(self.lpt_anbinmin, self.lpt_anbinmax)), 'd')
        self.bins_analysis = np.array(list(zip(self.lpt_finbinmin, self.lpt_finbinmax)), 'd')
        bin_matching = [
            [ptrange[0] <= bin[0] and ptrange[1] >= bin[1] for ptrange in self.bins_skimming].index(True)
            for bin in self.bins_analysis
        ]

        # Make it backwards-compatible
        if not self.lpt_probcutfin:
            lpt_probcutfin_tmp = datap["mlapplication"]["probcutoptimal"]
            self.lpt_probcutfin = []
            for i in range(self.p_nptfinbins):
                bin_id = bin_matching[i]
                self.lpt_probcutfin.append(lpt_probcutfin_tmp[bin_id])

        if self.mltype == "MultiClassification":
            for probcutfin, probcutpre in zip(self.lpt_probcutfin, self.lpt_probcutpre):
                if probcutfin[0] > probcutpre[0] or probcutfin[1] < probcutpre[1] or probcutfin[2] < probcutpre[2]:
                    self.logger.fatal("Probability cut final: %s must be tighter than presel %s!\n" \
                            "Verify that bkg prob presel > final, and other cuts presel < final",
                            self.lpt_probcutfin, self.lpt_probcutpre)
        elif self.lpt_probcutfin < self.lpt_probcutpre:
            self.logger.fatal("Probability cut final: %s must be tighter (smaller values) than presel %s!",
                    self.lpt_probcutfin, self.lpt_probcutpre)

        if self.mltype == "MultiClassification":
            self.l_selml = []
            comps = ["<=", ">=", ">="]
            for ipt in range(self.p_nptfinbins):
                mlsel_multi = [f'y_test_prob{self.p_modelname}{label.replace("-", "_")} ' \
                               f'{comp} {probcut}'
                               for label, comp, probcut in zip(self.class_labels, comps,
                                                               self.lpt_probcutfin[ipt])]
                self.l_selml.append(" and ".join(mlsel_multi))

        else:
            self.l_selml = [f"y_test_prob{self.p_modelname} > {self.lpt_probcutfin[ipt]}" \
                           for ipt in range(self.p_nptfinbins)]

        self.d_pkl_dec = d_pkl_dec
        self.mptfiles_recosk = []
        self.mptfiles_gensk = []

        self.d_pkl_decmerged = d_pkl_decmerged
        self.n_filemass = os.path.join(self.d_results, self.n_filemass)
        self.n_fileeff = os.path.join(self.d_results, self.n_fileeff)
        self.n_fileresp = os.path.join(self.d_results, self.n_fileresp)

        self.lpt_recosk = [self.n_reco.replace(".p", "_%s%d_%d.p" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_gensk = [self.n_gen.replace(".p", "_%s%d_%d.p" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_reco_ml = [os.path.join(self.d_pkl_ml, self.lpt_recosk[ipt]) \
                             for ipt in range(self.p_nptbins)]
        self.lpt_gen_ml = [os.path.join(self.d_pkl_ml, self.lpt_gensk[ipt]) \
                            for ipt in range(self.p_nptbins)]
        self.f_evt_count_ml = os.path.join(self.d_pkl_ml, self.n_evt_count_ml)
        self.lpt_recodec = None
        if self.doml is True:
            if self.mltype == "MultiClassification":
                self.lpt_recodec = [self.n_reco.replace(".p", "%d_%d_%.2f%.2f%.2f.p" % \
                                   (self.lpt_anbinmin[i], self.lpt_anbinmax[i],
                                    self.lpt_probcutpre[i][0], self.lpt_probcutpre[i][1],
                                    self.lpt_probcutpre[i][2])) \
                                    for i in range(self.p_nptbins)]
            else:
                self.lpt_recodec = [self.n_reco.replace(".p", "%d_%d_%.2f.p" % \
                                   (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                                    self.lpt_probcutpre[i])) for i in range(self.p_nptbins)]
        else:
            self.lpt_recodec = [self.n_reco.replace(".p", "%d_%d_std.p" % \
                               (self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                                                    for i in range(self.p_nptbins)]

        self.mptfiles_recosk = [createlist(self.d_pklsk, self.l_path, \
                                self.lpt_recosk[ipt]) for ipt in range(self.p_nptbins)]
        self.mptfiles_recoskmldec = [createlist(self.d_pkl_dec, self.l_path, \
                                   self.lpt_recodec[ipt]) for ipt in range(self.p_nptbins)]
        self.lpt_recodecmerged = [os.path.join(self.d_pkl_decmerged, self.lpt_recodec[ipt])
                                  for ipt in range(self.p_nptbins)]
        if self.mcordata == "mc":
            self.mptfiles_gensk = [createlist(self.d_pklsk, self.l_path, \
                                    self.lpt_gensk[ipt]) for ipt in range(self.p_nptbins)]
            self.lpt_gendecmerged = [os.path.join(self.d_pkl_decmerged, self.lpt_gensk[ipt])
                                     for ipt in range(self.p_nptbins)]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger

 #       if os.path.exists(self.d_root) is False:
 #           self.logger.warning("ROOT tree folder is not there. Is it intentional?")

        # Analysis cuts (loaded in self.process_histomass)
        self.analysis_cuts = None
        # Flag if they should be used
        self.do_custom_analysis_cuts = datap["analysis"][self.typean].get("use_cuts", False)

    def cfg(self, param, default = None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
                      param.split("."), self.datap['analysis'][self.typean])

    def unpack(self, file_index, max_no_keys = None): # pylint: disable=too-many-branches
        def dfread(rdir, trees, cols, idx_name=None):
            """Read DF from multiple (joinable) O2 tables"""
            try:
                if not isinstance(trees, list):
                    trees = [trees]
                    cols = [cols]
                # if all(type(var) is str for var in vars): vars = [vars]
                df = None
                for tree, col in zip([rdir[name] for name in trees], cols):
                    try:
                        data = tree.arrays(expressions=col, library='np')
                        dfnew = pd.DataFrame(columns=col, data=data)
                        df = pd.concat([df, dfnew], axis=1)
                    except Exception as e: # pylint: disable=broad-except
                        tree.show()
                        self.logger.critical('Failed to read data frame from tree %s: %s',
                                             tree.name, str(e))
                        sys.exit()
                df['df'] = int(df_no)
                if idx_name:
                    # df.rename_axis(idx_name, inplace=True)
                    df[idx_name] = df.index
                    df.set_index(['df', idx_name], inplace=True)
                return df
            except Exception as e:
                self.logger.exception('Failed to read data from trees: %s', str(e))
                raise e

        def dfappend(name: str, dfa):
            """Append DF row-wise"""
            dfs[name] = pd.concat([dfs.get(name, None), dfa])

        def dfmerge(dfl, dfr, **kwargs):
            """Merge dfl and dfr"""
            try:
                return pd.merge(dfl, dfr, **kwargs)
            except Exception as e:
                self.logger.error('merging failed: %s', str(e))
                dfl.info()
                dfr.info()
                raise e

        def dfuse(df_spec):
            level = df_spec.get('level', 'all')
            return ((level == 'all') or
                    (level in ('mc', 'gen', 'det') and self.mcordata == 'mc') or
                    (level in ('data') and self.mcordata == 'data'))

        self.logger.info('unpacking: %s', self.l_root[file_index])
        dfs = {}
        self.logger.debug(' -> reading')
        with uproot.open(self.l_root[file_index]) as rfile:
            df_processed = set()
            keys = rfile.keys(recursive=False, filter_name='DF_*')
            self.logger.info('found %d dataframes, reading %s', len(keys), max_no_keys or "all")
            for (idx, key) in enumerate(keys[:max_no_keys]):
                if not (df_key := re.match('^DF_(\\d+);', key)):
                    continue
                if (df_no := int(df_key.group(1))) in df_processed:
                    self.logger.warning('multiple versions of DF %d', df_no)
                    continue
                self.logger.debug('processing DF %d - %d / %d', df_no, idx, len(keys))
                df_processed.add(df_no)
                rdir = rfile[key]

                for df_name, df_spec in self.df_read.items():
                    if dfuse(df_spec):
                        trees = []
                        cols = []
                        for tree, spec in zip(df_spec['trees'].keys(), df_spec['trees'].values()):
                            if isinstance(spec, list):
                                trees.append(tree)
                                cols.append(spec)
                            elif dfuse(spec):
                                trees.append(tree)
                                cols.append(spec['vars'])
                        df = dfread(rdir, trees, cols, idx_name=df_spec.get('index', None))
                        dfappend(df_name, df)

        for df_name, df_spec in self.df_read.items():
            if dfuse(df_spec) and not dfs[df_name].empty:
                if 'extra' in df_spec:
                    self.logger.debug(' %s -> extra', df_name)
                    for col_name, col_val in df_spec['extra'].items():
                        dfs[df_name][col_name] = dfs[df_name].eval(col_val)
                if 'extract_component' in df_spec:
                    self.logger.debug(' %s -> extract_component', df_name)
                    specs = df_spec['extract_component']
                    for spec in specs:
                        var, newvar, component = spec['var'], spec['newvar'], spec['component']
                        dfs[df_name][newvar] = dfs[df_name][var].apply(lambda x, comp=component: x[comp])
                if 'filter' in df_spec:
                    self.logger.debug(' %s -> filter', df_name)
                    dfquery(dfs[df_name], df_spec['filter'], inplace=True)
                if 'tags' in df_spec:
                    self.logger.debug(' %s -> tags', df_name)
                    for tag, value in df_spec['tags'].items():
                        if dfuse(value):
                            dfs[df_name][tag] = np.array(
                                tag_bit_df(dfs[df_name], value['var'], value['req'], value.get('abs', False)),
                                dtype=int)
                if 'swap' in df_spec:
                    self.logger.debug(' %s -> swap', df_name)
                    spec = df_spec['swap']
                    if dfuse(spec):
                        swapped = dfs[df_name][spec['cand']] == dfs[df_name][spec['var_swap']] + 1
                        for var in spec['vars']:
                            dfs[df_name][var] = np.logical_and(dfs[df_name][var] == 1, swapped)
                self.logger.debug(' %s -> done', df_name)


        if self.df_merge:
            for m_spec in self.df_merge:
                base = m_spec['base']
                ref = m_spec['ref']
                out = m_spec.get('out', base)
                if all([dfuse(self.df_read[base]), dfuse(self.df_read[ref])]):
                    if (on := m_spec.get('use', None)) is not None:
                        self.logger.info('merging %s with %s on %s into %s', base, ref, on, out)
                        if not isinstance(on, list) or 'df' not in on:
                            on = ['df', on]
                        dfs[out] = dfmerge(dfs[base], dfs[ref], on=on)
                    elif (on := m_spec.get('left_on', None)) is not None:
                        self.logger.info('merging %s with %s on %s into %s', base, ref, on, out)
                        if not is_numeric_dtype(dfs[base][on]):
                            self.logger.info('exploding dataframe %s on variable %s', base, on)
                            dfs[base] = dfs[base].explode(on)
                        dfs[out] = dfmerge(dfs[base], dfs[ref], left_on=['df', on], right_index=True)
                    else:
                        var = self.df_read[ref]['index']
                        self.logger.info('merging %s with %s on %s (default) into %s', base, ref, var, out)
                        dfs[out] = dfmerge(dfs[base], dfs[ref], left_on=['df', var], right_index=True)
                    if 'extra' in m_spec:
                        self.logger.debug(' %s -> extra', out)
                        for col_name, col_val in m_spec['extra'].items():
                            dfs[out][col_name] = dfs[out].eval(col_val)

        if self.df_write:
            for df_name, df_spec in self.df_write.items():
                if dfuse(df_spec):
                    self.logger.info('writing %s to %s', df_name, df_spec['file'])
                    src = df_spec.get('source', df_name)
                    dfo = dfquery(dfs[src], df_spec.get('filter', None))
                    path = os.path.join(self.d_pkl, self.l_path[file_index], df_spec['file'])
                    write_df(dfo, path)

    def skim(self, file_index):
        dfreco = read_df(self.l_reco[file_index])
        dfgen = read_df(self.l_gen[file_index]) if self.mcordata == 'mc' else None

        for ipt in range(self.p_nptbins):
            dfrecosk = seldf_singlevar(dfreco, self.v_var_binning,
                                       self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt])
            dfrecosk = dfquery(dfrecosk, self.s_reco_skim[ipt])
            write_df(dfrecosk, self.mptfiles_recosk[ipt][file_index])

            if dfgen is not None:
                dfgensk = seldf_singlevar(dfgen, self.v_var_binning,
                                          self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt])
                dfgensk = dfquery(dfgensk, self.s_gen_skim[ipt])
                write_df(dfgensk, self.mptfiles_gensk[ipt][file_index])

    def applymodel(self, file_index):
        from machine_learning_hep.models import \
            apply  # pylint: disable=import-error, import-outside-toplevel
        for ipt in range(self.p_nptbins):
            if os.path.exists(self.mptfiles_recoskmldec[ipt][file_index]):
                if os.stat(self.mptfiles_recoskmldec[ipt][file_index]).st_size != 0:
                    continue
            dfrecosk = read_df(self.mptfiles_recosk[ipt][file_index])
            if self.p_mask_values:
                mask_df(dfrecosk, self.p_mask_values)
            if self.doml is True:
                if os.path.isfile(self.lpt_model[ipt]) is False:
                    print("Model file not present in bin %d" % ipt)
                with openfile(self.lpt_model[ipt], 'rb') as mod_file:
                    mod = pickle.load(mod_file)
                if self.mltype == "MultiClassification":
                    dfrecoskml = apply(self.mltype, [self.p_modelname], [mod],
                                       dfrecosk, self.v_train[ipt], self.class_labels)
                    probs = [f'y_test_prob{self.p_modelname}{label.replace("-", "_")}' \
                             for label in self.class_labels]
                    dfrecoskml = dfrecoskml[(dfrecoskml[probs[0]] <= self.lpt_probcutpre[ipt][0]) &
                                            (dfrecoskml[probs[1]] >= self.lpt_probcutpre[ipt][1]) &
                                            (dfrecoskml[probs[2]] >= self.lpt_probcutpre[ipt][2])]
                else:
                    dfrecoskml = apply("BinaryClassification", [self.p_modelname], [mod],
                                       dfrecosk, self.v_train[ipt])
                    probvar = f"y_test_prob{self.p_modelname}"
                    dfrecoskml = dfrecoskml.loc[dfrecoskml[probvar] > self.lpt_probcutpre[ipt]]
            else:
                dfrecoskml = dfrecosk.query("isstd == 1")
            write_df(dfrecoskml, self.mptfiles_recoskmldec[ipt][file_index])

    @staticmethod
    def callback(ex):
        get_logger().exception('Error callback: %s', ex)
        raise ex

    def parallelizer(self, function, argument_list, maxperchunk):
        # TODO: fix logic and avoid waiting for the slowest job
        chunks = [argument_list[x:x+maxperchunk]
                  for x in range(0, len(argument_list), maxperchunk)]
        for chunk in chunks:
            self.logger.debug("Processing new chunk of size = %i", maxperchunk)
            with mp.Pool(self.p_maxprocess) as pool:
                _ = [pool.apply_async(function, args=chunk[i], error_callback=self.callback)
                     for i in range(len(chunk))]
                pool.close()
                pool.join()
                # TODO: maybe simpler to use:
                # for _ in pool.imap_unordered(function, (x[0] for x in chunk)):
                #     pass

    def process_unpack_par(self):
        self.logger.info("Unpacking %s period %s", self.mcordata, self.period)
        create_folder_struc(self.d_pkl, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.logger.debug('d_pkl: %s, l_path: %s, arguments: %s',
                          self.d_pkl, str(self.l_path), str(arguments))
        self.parallelizer(self.unpack, arguments, self.p_chunksizeunp)

    def process_skim_par(self):
        self.logger.info("Skimming %s period %s", self.mcordata, self.period)
        create_folder_struc(self.d_pklsk, self.l_path)
        arguments = [(i,) for i in range(len(self.l_reco))]
        self.parallelizer(self.skim, arguments, self.p_chunksizeskim)
        if self.p_dofullevtmerge is True:
            merge_method(self.l_evt, self.f_totevt)
            merge_method(self.l_evtorig, self.f_totevtorig)

    def process_applymodel_par(self):
        self.logger.info("Applying model to %s period %s", self.mcordata, self.period)
        create_folder_struc(self.d_pkl_dec, self.l_path)
        arguments = [(i,) for i in range(len(self.mptfiles_recosk[0]))]
        self.parallelizer(self.applymodel, arguments, self.p_chunksizeskim)

    def process_mergeforml(self):
        self.logger.info("doing merging for ml %s %s", self.mcordata, self.period)
        indices_for_evt = []
        for ipt in range(self.p_nptbins):
            nfiles = len(self.mptfiles_recosk[ipt])
            if not nfiles:
                print("There are no files to be merged")
                sys.exit(1)
            self.logger.info("Use merge fraction %g for pT bin %d",
                             self.p_frac_merge[ipt], ipt)
            ntomerge = int(nfiles * self.p_frac_merge[ipt])
            rd.seed(self.p_rd_merge)
            filesel = rd.sample(range(0, nfiles), ntomerge)
            indices_for_evt = list(set(indices_for_evt) | set(filesel))
            list_sel_recosk = [self.mptfiles_recosk[ipt][j] for j in filesel]
            merge_method(list_sel_recosk, self.lpt_reco_ml[ipt])
            if self.mcordata == "mc":
                list_sel_gensk = [self.mptfiles_gensk[ipt][j] for j in filesel]
                merge_method(list_sel_gensk, self.lpt_gen_ml[ipt])

        self.logger.info("Count events...")
        list_sel_evt = [self.l_evt[j] for j in indices_for_evt]
        list_sel_evtorig = [self.l_evtorig[j] for j in indices_for_evt]
        count_dict = {"evt": count_df_length_pkl(*list_sel_evt),
                      "evtorig": count_df_length_pkl(*list_sel_evtorig)}
        dump_yaml_from_dict(count_dict, self.f_evt_count_ml)

    def process_mergedec(self):
        for ipt in range(self.p_nptbins):
            merge_method(self.mptfiles_recoskmldec[ipt], self.lpt_recodecmerged[ipt])
            if self.mcordata == "mc":
                merge_method(self.mptfiles_gensk[ipt], self.lpt_gendecmerged[ipt])


    def load_cuts(self):
        """Load cuts from database
        """

        # Assume that there is a list with self.p
        raw_cuts = self.datap["analysis"][self.typean].get("cuts", None)
        if not raw_cuts:
            print("No custom cuts given, hence not cutting...")
            self.analysis_cuts = [None] * self.p_nptfinbins
            return

        if len(raw_cuts) != self.p_nptfinbins:
            print(f"You have {self.p_nptfinbins} but you passed {len(raw_cuts)} cuts. Exit...")
            sys.exit(1)

        self.analysis_cuts = deepcopy(raw_cuts)


    def apply_cuts_ptbin(self, df_, ipt):
        """Helper function to cut dataframe with cuts for given pT bin

        Args:
            df: dataframe
            ipt: int
                i'th pT bin
        Returns:
            dataframe
        """
        if not self.analysis_cuts[ipt]:
            return df_

        return df_.query(self.analysis_cuts[ipt])


    def process_histomass(self):
        self.logger.debug("Doing masshisto %s %s", self.mcordata, self.period)
        self.logger.debug("Using run selection for mass histo %s %s %s",
                          self.runlistrigger, "for period", self.period)
        if self.doml is True:
            self.logger.debug("Doing ml analysis")
        elif self.do_custom_analysis_cuts:
            self.logger.debug("Using custom cuts")
        else:
            self.logger.debug("No extra selection needed since we are doing std analysis")

        # Load potential custom cuts
        self.load_cuts()

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_histomass_single, arguments, self.p_chunksizeunp) # pylint: disable=no-member
        with tempfile.TemporaryDirectory() as tmp_merged_dir:
            mergerootfiles(self.l_histomass, self.n_filemass, tmp_merged_dir)

    def process_efficiency(self):
        print("Doing efficiencies", self.mcordata, self.period)
        print("Using run selection for eff histo", \
               self.runlistrigger, "for period", self.period)
        if self.doml is True:
            print("Doing ml analysis")
        elif self.do_custom_analysis_cuts:
            print("Using custom cuts")
        else:
            print("No extra selection needed since we are doing std analysis")

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_efficiency_single, arguments, self.p_chunksizeunp) # pylint: disable=no-member
        with tempfile.TemporaryDirectory() as tmp_merged_dir:
            mergerootfiles(self.l_histoeff, self.n_fileeff, tmp_merged_dir)
