#!/bin/env python3

#  © Copyright CERN 2025. All rights not expressly granted are reserved.  #
# This program is free software: you can redistribute it and/or modify it #
#  under the terms of the GNU General Public License as published by the  #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version. This program is distributed in the hope that #
#  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  #
#     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#           See the GNU General Public License for more details.          #
#    You should have received a copy of the GNU General Public License    #
#   along with this program. if not, see <https://www.gnu.org/licenses/>. #

"""
Plot different results for a given observable together.
Author: Vit Kucera <vit.kucera@cern.ch>

Produces one plot per particle per observable and also one plot per observable with all particles.
"""

import sys

from ROOT import TFile, gROOT

from machine_learning_hep.utilities import get_colour, make_plot

config = {
    "particles": {
        "d0": {
            "activate": 1,
            "label": "D^{0}",
        },
        "lc": {
            "activate": 1,
            "label": "#Lambda^{#plus}_{c}",
        },
        "dplus": {
            "activate": 1,
            "label": "D^{#plus}",
        },
        "incl": {
            "activate": 1,
            "label": "inclusive",
        },
    },
    "observables": {
        "zg": {
            "activate": 1,
            "range": [0.1, 0.5],
            "label": "#it{z}_{g}",
            "leg": [0.7, 0.65, 0.85, 0.85],
        },
        "rg": {
            "activate": 1,
            "range": [0.0, 0.4],
            "label": "#it{R}_{g}",
            "leg": [0.15, 0.65, 0.4, 0.85],
        },
        "nsd": {
            "activate": 1,
            "range": [-0.5, 5.5],
            "label": "#it{n}_{SD}",
            "leg": [0.7, 0.65, 0.85, 0.85],
        },
        "zpar": {
            "activate": 1,
            "range": [0.0, 1.0],
            "label": "#it{z}_{#parallel}",
            "leg": [0.15, 0.65, 0.4, 0.85],
        },
    },
    "results": {
        "d0": {
            "run2": {
                "label": "Run 2",
                "activate": 1,
                "path_file": "/data2/MLhep/results_run2.root",
                "colour": -1,
                "name_hist": {
                    "zg": "zg_hf_data_1_stat",
                    "rg": "rg_hf_data_1_stat",
                    "nsd": "nsd_hf_data_1_stat",
                },
            },
            "hp24": {
                "label": "HP24",
                "activate": 1,
                "path_file": "/home/vkucera/hp24/d0/results.root",
                "colour": 0,
                "name_hist": {
                    "zg": "h_zg_sidesub_unfolded_data_ptjet-15-30_sel_selfnorm",
                    "rg": "h_rg_sidesub_unfolded_data_ptjet-15-30_sel_selfnorm",
                    "nsd": "h_nsd_sidesub_unfolded_data_ptjet-15-30_sel_selfnorm",
                    "zpar": "h_zpar_sidesub_unfolded_data_ptjet-15-30_sel_selfnorm",
                },
            },
            "qm25_prv": {
                "label": "QM25 preview",
                "activate": 0,
                "path_file": "/home/vkucera/mlhep/d0jet/jet_obs_qm25_preview_02-13/"
                "default/default/data/results_all/results.root",
                "colour": 1,
                "name_hist": {
                    "zg": "h_zg_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "rg": "h_rg_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "nsd": "h_nsd_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "zpar": "h_zpar_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                },
            },
            "qm25_pwg": {
                "label": "QM25 PWG",
                "activate": 1,
                "path_file": "/home/vkucera/mlhep/d0jet/jet_obs_qm25_pwg_03-20/"
                "default/default/data/results_all/results.root",
                "colour": 2,
                "name_hist": {
                    "zg": "h_zg_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "rg": "h_rg_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "nsd": "h_nsd_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "zpar": "h_zpar_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                },
            },
        },
        "lc": {
            "run2": {
                "label": "Run 2",
                "activate": 1,
                "path_file": "/home/vkucera/mlhep/run2/results/lc/unfolding_results.root",
                "colour": -1,
                "name_hist": {
                    "zpar": "unfolded_z_sel_pt_jet_7.00_15.00",
                },
            },
            "hp24": {
                "label": "HP24",
                "activate": 1,
                "path_file": "/home/vkucera/hp24/lc/results.root",
                "colour": 0,
                "name_hist": {
                    "zg": "h_zg_sidesub_unfolded_data_ptjet-15-30_sel_selfnorm",
                    "rg": "h_rg_sidesub_unfolded_data_ptjet-15-30_sel_selfnorm",
                    "nsd": "h_nsd_sidesub_unfolded_data_ptjet-15-30_sel_selfnorm",
                    "zpar": "h_zpar_sidesub_unfolded_data_ptjet-7-15_sel_selfnorm",
                },
            },
            "qm25_prv": {
                "label": "QM25 preview",
                "activate": 0,
                "path_file": "/home/vkucera/mlhep/lcjet/jet_obs_nbkp/default/default/data/results_all/results.root",
                "colour": 1,
                "name_hist": {
                    "zpar": "h_zpar_sidesub_unfolded_data_ptjet-7-10_sel_selfnorm",
                },
            },
            "qm25_pwg": {
                "label": "QM25 PWG",
                "activate": 1,
                "path_file": "/home/ldellost/mlhep/lcjet_Crystal/jet_obs/"
                "default/default/data/results_all_Jochen/results.root",
                "colour": 2,
                "name_hist": {
                    "zpar": "h_zpar_sidesub_unfolded_data_ptjet-7-15_sel_selfnorm",
                },
            },
        },
        "dplus": {
            "qm25_pwg": {
                "label": "QM25 PWG",
                "activate": 1,
                "path_file": "/home/jklein/mlhep/dpjet2/jet_obs/default/default/data/results_all/results.root",
                "colour": 2,
                "name_hist": {
                    "zg": "h_zg_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "rg": "h_rg_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "nsd": "h_nsd_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                    "zpar": "h_zpar_sidesub_unfolded_data_ptjet-15-20_sel_selfnorm",
                },
            },
        },
        "incl": {
            "run2": {
                "label": "Run 2",
                "activate": 1,
                "path_file": "/home/vkucera/mlhep/run2/results/d0/results_all.root",
                "colour": -1,
                "name_hist": {
                    "zg": "zg_hf_data_1_stat",
                    "rg": "rg_hf_data_1_stat",
                    "nsd": "nsd_hf_data_1_stat",
                },
            },
        },
    },
}

gROOT.SetBatch(True)
DIR_OUTPUT = "."

particles = [particle for particle, cfg in config["particles"].items() if cfg["activate"]]
print(f"Particles: {particles}")

observables = [obs for obs, cfg in config["observables"].items() if cfg["activate"]]
print(f"Observables: {observables}")

datasets = {}
for particle in particles:
    print(f"\nProcessing particle {particle}")
    cfg_results = config["results"][particle]
    datasets[particle] = [ds for ds, cfg in cfg_results.items() if cfg["activate"]]
    print(f"Datasets: {datasets[particle]}")
    for ds in datasets[particle]:
        path_file = cfg_results[ds]["path_file"]
        print(f"Opening file {ds}:{path_file}")
        if not (file := TFile.Open(path_file)):
            print("Failed")
            sys.exit(1)
        cfg_results[ds]["file"] = file

for obs in observables:
    print(f"\nProcessing observable {obs}")
    histograms_obs = []
    labels_obs = []
    for particle in particles:
        print(f"\nProcessing particle {particle}")
        histograms = []
        labels = []
        colours = []
        for ds in datasets[particle]:
            cfg_results = config["results"][particle][ds]
            if obs not in cfg_results["name_hist"]:
                print(f"Skipping {ds}")
                continue
            name_hist = cfg_results["name_hist"][obs]
            print(f"Getting histogram {ds}:{name_hist}")
            if not (hist := cfg_results["file"].Get(name_hist)):
                print("Failed")
                sys.exit(1)
            histograms.append(hist)
            labels.append(cfg_results["label"])
            colours.append(get_colour(cfg_results["colour"]))
        make_plot(
            name=f"{particle}_{obs}",
            title=f"{config['particles'][particle]['label']}"
            f";{config['observables'][obs]['label']}"
            f";(1/#it{{N}}_{{jet}}) d#it{{N}}/d{config['observables'][obs]['label']}",
            list_obj=histograms,
            labels_obj=labels,
            colours=colours,
            leg_pos=config["observables"][obs]["leg"],
            range_x=config["observables"][obs]["range"],
            path=DIR_OUTPUT,
        )
        histograms_obs += histograms
        labels_obs += [f"{config['particles'][particle]['label']}: {lab}" for lab in labels]
    print("\nProcessing all particles")
    make_plot(
        name=f"all_{obs}",
        title=f"all particles"
        f";{config['observables'][obs]['label']}"
        f";(1/#it{{N}}_{{jet}}) d#it{{N}}/d{config['observables'][obs]['label']}",
        list_obj=histograms_obs,
        labels_obj=labels_obs,
        leg_pos=config["observables"][obs]["leg"],
        range_x=config["observables"][obs]["range"],
        path=DIR_OUTPUT,
    )
