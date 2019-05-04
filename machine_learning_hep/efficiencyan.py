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
"""
Methods to: study expected efficiency
"""
import os.path
import array
import math
import pandas as pd
import yaml
from ROOT import TFile, TH1F # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.general import filter_df_cand
from machine_learning_hep.general import filterdataframe_singlevar
from machine_learning_hep.selectionutils import selectcand_lincut

# pylint: disable=too-many-statements, too-many-locals, too-many-branches
def analysis_eff(run_config, data_dict, case, sel_type, index):

    analysis_bin_min = run_config['analysis']['analysisbinmin']
    analysis_bin_max = run_config['analysis']['analysisbinmax']
    model_bin_min = run_config['analysis']['modelbinmin']
    model_bin_max = run_config['analysis']['modelbinmax']
    modeltouse = run_config['analysis']['modeltouse']
    model_name_temp = run_config['analysis']['modelname']
    cuts = run_config['analysis']['probcutoptimal']

    data_dict = data_dict[case]
    var_bin = data_dict['variables']['var_binning']
    folder_mc = data_dict['output_folders']['pkl_final']['mc'][index]
    out_dir = data_dict['output_folders']['plotsanalysis']["mc"][index]
    file_mc_reco = data_dict['files_names']['namefile_reco_skim_ml_tot']
    file_mc_gen = data_dict['files_names']['namefile_gen_skim_tot']
    usecustomsel = run_config["analysis"]["mc"]["std"]["usecustom"]


    cuts_map = None
    if sel_type == 'std':
        file_mc_reco = data_dict['files_names']['namefile_reco_skim_std_tot']
        if usecustomsel:
            cuts_config_filename = data_dict["custom_std_sel"]["cuts_config_file"]
            with open(cuts_config_filename, 'r') as cuts_config:
                cuts_map = yaml.load(cuts_config)
            #NB: in case of custom linear selections it overrides pT bins of default_complete
            analysis_bin_min = cuts_map["var_binning"]["min"]
            analysis_bin_max = cuts_map["var_binning"]["max"]

    n_bins = len(analysis_bin_min)
    analysis_bin_lims_temp = list(analysis_bin_min)
    analysis_bin_lims_temp.append(analysis_bin_max[n_bins-1])
    analysis_bin_lims = array.array('f', analysis_bin_lims_temp)


    h_gen_pr = TH1F("h_gen_pr", "Prompt Generated in acceptance |y|<0.5", \
                    n_bins, analysis_bin_lims)
    h_presel_pr = TH1F("h_presel_pr", "Prompt Reco in acc |#eta|<0.8 and sel", \
                       n_bins, analysis_bin_lims)
    h_sel_pr = TH1F("h_sel_pr", "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                    n_bins, analysis_bin_lims)
    h_gen_fd = TH1F("h_gen_fd", "FD Generated in acceptance |y|<0.5", \
                    n_bins, analysis_bin_lims)
    h_presel_fd = TH1F("h_presel_fd", "FD Reco in acc |#eta|<0.8 and sel", \
                       n_bins, analysis_bin_lims)
    h_sel_fd = TH1F("h_sel_fd", "FD Reco and sel in acc |#eta|<0.8 and sel", \
                    n_bins, analysis_bin_lims)

    bincounter = 0

    for b_min, b_max in zip(analysis_bin_min, analysis_bin_max):

        file_mc_reco_bin = file_mc_reco
        file_mc_gen_bin = file_mc_gen
        if sel_type == "ml":
            binsamplemin = model_bin_min[modeltouse[bincounter]]
            binsamplemax = model_bin_max[modeltouse[bincounter]]
            file_mc_reco_bin = file_mc_reco_bin.replace(".pkl", "%d_%d.pkl" \
                                                        % (binsamplemin, binsamplemax))
            print(folder_mc + "/" + file_mc_reco_bin)

        df_mc_reco = pd.read_pickle(os.path.join(folder_mc, file_mc_reco_bin))
        df_mc_reco = df_mc_reco.query(data_dict['presel_reco']) #probably not necessary any more
        df_mc_gen = pd.read_pickle(os.path.join(folder_mc, file_mc_gen_bin))
        df_mc_gen = df_mc_gen.query(data_dict['presel_gen'])

        df_reco_sel = filterdataframe_singlevar(df_mc_reco, var_bin, b_min, b_max)
        df_gen_sel = filterdataframe_singlevar(df_mc_gen, var_bin, b_min, b_max)
        print("analysis bin min=", b_min, ",bin max=", b_max)

        df_gen_sel_pr = filter_df_cand(df_gen_sel, data_dict, 'mc_signal_prompt')
        df_reco_presel_pr = filter_df_cand(df_reco_sel, data_dict, 'mc_signal_prompt')
        df_reco_sel_pr = df_reco_presel_pr
        if sel_type == "std":
            if usecustomsel:
                df_reco_sel_pr = filter_df_cand(df_reco_sel_pr, data_dict, 'presel_track_pid')
            #apply standard cuts from file
                for _, icutvar in cuts_map.items():
                    if not df_reco_sel_pr.empty and icutvar["name"] != "var_binning":
                        array_var = df_reco_sel_pr.loc[:, icutvar["name"]].values
                        is_selected = selectcand_lincut(array_var, icutvar["min"][bincounter], \
                            icutvar["max"][bincounter], icutvar["isabsval"])
                        df_reco_sel_pr = df_reco_sel_pr[is_selected]

            else:
                df_reco_sel_pr = filter_df_cand(df_reco_sel_pr, data_dict, 'sel_std_analysis')

        if sel_type == "ml":
            df_reco_sel_pr = df_reco_sel_pr[df_reco_sel_pr['y_test_prob' + \
                                            model_name_temp].values >= cuts[modeltouse[bincounter]]]

        h_gen_pr.SetBinContent(bincounter + 1, len(df_gen_sel_pr))
        h_gen_pr.SetBinError(bincounter + 1, math.sqrt(len(df_gen_sel_pr)))
        h_presel_pr.SetBinContent(bincounter + 1, len(df_reco_presel_pr))
        h_presel_pr.SetBinError(bincounter + 1, math.sqrt(len(df_reco_presel_pr)))
        h_sel_pr.SetBinContent(bincounter + 1, len(df_reco_sel_pr))
        h_sel_pr.SetBinError(bincounter + 1, math.sqrt(len(df_reco_sel_pr)))
        print("prompt efficiency tot ptbin=", bincounter, ", value = ",
              len(df_reco_sel_pr)/len(df_gen_sel_pr))

        df_gen_sel_fd = filter_df_cand(df_gen_sel, data_dict, 'mc_signal_FD')
        df_reco_presel_fd = filter_df_cand(df_reco_sel, data_dict, 'mc_signal_FD')
        df_reco_sel_fd = df_reco_presel_fd
        if sel_type == "std":
            if usecustomsel:
                df_reco_sel_fd = filter_df_cand(df_reco_sel_fd, data_dict, 'presel_track_pid')
            #apply standard cuts from file
                for _, icutvar in cuts_map.items():
                    if not df_reco_sel_fd.empty and icutvar["name"] != "var_binning":
                        array_var = df_reco_sel_fd.loc[:, icutvar["name"]].values
                        is_selected = selectcand_lincut(array_var, icutvar["min"][bincounter], \
                            icutvar["max"][bincounter], icutvar["isabsval"])
                        df_reco_sel_fd = df_reco_sel_fd[is_selected]
            else:
                df_reco_sel_fd = filter_df_cand(df_reco_sel_fd, data_dict, 'sel_std_analysis')

        if sel_type == "ml":
            df_reco_sel_fd = df_reco_sel_fd[df_reco_sel_fd['y_test_prob' + \
                                            model_name_temp].values >= cuts[modeltouse[bincounter]]]

        h_gen_fd.SetBinContent(bincounter + 1, len(df_gen_sel_fd))
        h_gen_fd.SetBinError(bincounter + 1, math.sqrt(len(df_gen_sel_fd)))
        h_presel_fd.SetBinContent(bincounter + 1, len(df_reco_presel_fd))
        h_presel_fd.SetBinError(bincounter + 1, math.sqrt(len(df_reco_presel_fd)))
        h_sel_fd.SetBinContent(bincounter + 1, len(df_reco_sel_fd))
        h_sel_fd.SetBinError(bincounter + 1, math.sqrt(len(df_reco_sel_fd)))
        print("fd efficiency tot ptbin=", bincounter, ", value = ",
              len(df_reco_sel_fd)/len(df_gen_sel_fd))
        bincounter = bincounter + 1
    usecustomsel = (int)(usecustomsel)
    out_file = TFile.Open( \
               f'{out_dir}/efficiencies_{case}_{sel_type}_custom{usecustomsel}.root', 'recreate')
    out_file.cd()
    h_gen_pr.Write()
    h_presel_pr.Write()
    h_sel_pr.Write()
    h_gen_fd.Write()
    h_presel_fd.Write()
    h_sel_fd.Write()
