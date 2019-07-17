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

import argparse
from machine_learning_hep.logger import configure_logger, get_logger
from machine_learning_hep.io import parse_yaml
from machine_learning_hep.plotting import process_files, plot_from_yamls, save_plots

def process_pickles(args):
    logger = get_logger()
    plot_config = parse_yaml(args.plot_config)

    logger_string = f"Processing files with plotting configuration {args.plot_config}.\nPlots will be written to {args.output_dir}."
    logger.info(logger_string)
    names, axes, stats = process_files(args.top, args.file_signature, args.recursive, args.n_files, plot_config)
    save_plots(names, axes, stats, args.output_dir)

def process_yamls(args):
    logger = get_logger()

    logger_string = f"Processing YAML files. Plots will be written to {args.output_dir}."
    logger.info(logger_string)
    names, axes, stats = plot_from_yamls(args.top)
    save_plots(names, axes, stats, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="activate debug log level")
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_pickles = subparsers.add_parser('plot-pickles', help='pickle plots help')

    parser_pickles.add_argument("top", help="directory where it should be looked for ROOT files")
    parser_pickles.add_argument("--file-signature", dest="file_signature", help="regular expression matching filenames", default=r"AnalysisResultsGen.pkl")
    parser_pickles.add_argument("--recursive", action="store_true", help="recursive search for files")
    parser_pickles.add_argument("--n-files", dest="n_files", help="number of files to be processed", type=int, default=-1)
    parser_pickles.add_argument("--plot-config", dest="plot_config", help="plots to be produced", required=True)
    parser_pickles.add_argument("--output-dir", dest="output_dir", help="directory where plots are dumped", default="./plots_output/")
    parser_pickles.set_defaults(func=process_pickles)

    parser_yamls = subparsers.add_parser('plot-yamls', help='YAML plots help')
    parser_yamls.add_argument("top", help="directory where it should be looked for ROOT files", nargs="+")
    parser_yamls.add_argument("--output-dir", dest="output_dir", help="directory where plots are dumped", default="./plots_output/")
    parser_yamls.set_defaults(func=process_yamls)

    args = parser.parse_args()

    configure_logger(args.debug)

    args.func(args)


    #logger = get_logger()

