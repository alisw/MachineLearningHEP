#!/bin/env python3

"""This module downloads AO2Ds from an ALICE hyperloop train run"""

import argparse
import os
from pathlib import PurePosixPath

import requests  # pylint: disable=import-error

try:
    from alienpy import alien, xrd_core
except ImportError:
    print("Could not import alien, install with pip install alienpy")


def get_train_spec(train_id: int):
    """Retrieve train spec from hyperloop interface"""
    # https://alimonitor.cern.ch/hyperloop/train-run/131050
    URL = f"https://alimonitor.cern.ch/alihyperloop-data/trains/train.jsp?train_id={args.train_id}"
    try:
        return requests.get(
            URL,
            verify=False,
            cert=(f"/tmp/tokencert_{os.getuid()}.pem", f"/tmp/tokenkey_{os.getuid()}.pem"),
            timeout=10,
        )
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        raise


def find_ao2ds(a: alien.AliEn, dir: str) -> list[str]:
    """Find AO2Ds in train output directory"""
    CMD_FIND = f"find {PurePosixPath(dir) / 'AOD'} AO2D.root"
    ret = a.run(CMD_FIND)
    if ret.exitcode == 0:
        return ret.out.split()
    else:
        print(f"Failed to run search: {CMD_FIND}\n{ret.out}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AO2Ds from hyperloop train")
    parser.add_argument("train_id", type=int, help="train ID")
    parser.add_argument("--prefix", "-p", default="/data2/MLhep/trains/")
    parser.add_argument("--dry-run", "-n", action="store_true", help="dry run")
    args = parser.parse_args()

    train_spec = get_train_spec(args.train_id)
    outputdirs = [d["outputdir"] for d in train_spec.json()["jobResults"]]

    a = alien.AliEn()
    SRC = [dir for outputdir in outputdirs for dir in find_ao2ds(a, outputdir)]
    DST = ["file:" + str(PurePosixPath(args.prefix) / str(args.train_id) / file.lstrip("/")) for file in SRC]
    for s, d in zip(SRC, DST):
        print(f"Copying {s} to {d}")
    if not args.dry_run:
        xrd_core.DO_XrootdCp(a.wb(), api_src=SRC, api_dst=DST)
