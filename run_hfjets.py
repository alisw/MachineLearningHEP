#!/bin/env python3

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
    db = 'machine_learning_hep/data/data_run3/database_ml_parameters_D0pp_jet.yml'
elif args.case == 'lcjet':
    db = 'machine_learning_hep/data/data_run3/database_ml_parameters_LcJet_pp.yml'
else:
    print(f'Unknown case <{args.case}>')

for step in args.steps:
    subprocess.run(f'mlhep -r machine_learning_hep/submission/d0jet_{step}.yml ' +
                   f'-d {db} -a {args.analysis} {"--delete" if args.delete else ""}',
                   shell=True, stdout=sys.stdout, stderr=sys.stderr)
