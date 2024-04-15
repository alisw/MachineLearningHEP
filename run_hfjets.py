#!/bin/env python3
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

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--case', '-c', default='d0jet')
parser.add_argument('--analysis', '-a', default='jet_obs')
parser.add_argument('--steps', '-s', nargs='+', default=['ana'])
parser.add_argument('--delete', '-d', action='store_true')
# parser.add_argument('--dryrun', '-n', action='store_true')
args = parser.parse_args()

if args.case == 'd0jet':
    DB = 'machine_learning_hep/data/data_run3/database_ml_parameters_D0pp_jet.yml'
elif args.case == 'lcjet':
    DB = 'machine_learning_hep/data/data_run3/database_ml_parameters_LcJet_pp.yml'
else:
    print(f'Unknown case <{args.case}>')
    sys.exit(-1)

for step in args.steps:
    subprocess.run(f'mlhep -r machine_learning_hep/submission/d0jet_{step}.yml ' +
                   f'-d {DB} -a {args.analysis} {"--delete" if args.delete else ""}',
                   shell=True, stdout=sys.stdout, stderr=sys.stderr, check=True)
