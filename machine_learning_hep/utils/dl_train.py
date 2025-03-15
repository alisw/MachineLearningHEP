#!/bin/env python3

"""This script downloads AO2Ds from a hyperloop train"""

import argparse
import os
import subprocess
import sys

import requests  # pylint: disable=import-error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AO2Ds from hyperloop train")
    parser.add_argument("train_id", type=int, help="train ID")
    parser.add_argument("--dry-run", "-n", action="store_true", help="dry run")
    args = parser.parse_args()

    # https://alimonitor.cern.ch/hyperloop/train-run/131050
    URL = f"https://alimonitor.cern.ch/alihyperloop-data/trains/train.jsp?train_id={args.train_id}"
    try:
        train_spec = requests.get(
            URL,
            verify=False,
            cert=(f"/tmp/tokencert_{os.getuid()}.pem", f"/tmp/tokenkey_{os.getuid()}.pem"),
            timeout=10,
        )
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        sys.exit(1)
    outputdirs = [d["outputdir"] for d in train_spec.json()["jobResults"]]

    TBASE = f"/data2/MLhep/trains/{args.train_id}"
    SCRIPT = "/home/jklein/alisw.bak/Run3Analysisvalidation/exec/download_from_grid.sh"

    for outputdir in outputdirs:
        PATH = f"{outputdir}/AOD"
        CMD = f"{SCRIPT} {PATH} {TBASE}/{PATH} AO2D.root"
        if args.dry_run:
            print(f"Dry run: {CMD}")
        else:
            subprocess.run(CMD, shell=True, check=False, stdout=sys.stdout, stderr=sys.stderr)
