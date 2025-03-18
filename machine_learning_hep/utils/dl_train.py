#!/bin/env python3

"""This script downloads AO2Ds from an ALICE hyperloop train run"""

import argparse
import os
from pathlib import PurePosixPath
import sys

import requests  # pylint: disable=import-error

try:
    from alienpy import alien, xrd_core
except ImportError:
    print("Failed to import alien -> no alien support")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AO2Ds from hyperloop train")
    parser.add_argument("train_id", type=int, help="train ID")
    parser.add_argument("--prefix", "-p", default="/data2/MLhep/trains/")
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

    TBASE = PurePosixPath(args.prefix) / str(args.train_id)

    a = alien.AliEn()

    for outputdir in outputdirs:
        PATH = PurePosixPath(f"{outputdir}") / "AOD"
        CMD_FIND = f"find {PATH} AO2D.root"
        ret = a.run(CMD_FIND)
        if ret.exitcode == 0:
            SRC = ret.out.split()
            DST = ["file:" + str(PurePosixPath(TBASE) / file.lstrip("/")) for file in SRC]
            for s, d in zip(SRC, DST):
                print(f"Copying {s} to {d}")
            if not args.dry_run:
                xrd_core.DO_XrootdCp(a.wb(), api_src=SRC, api_dst=DST)
        else:
            print(f"Failed to run search: {CMD_FIND}\n{ret.out}")
            continue
