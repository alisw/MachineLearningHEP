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
    url = f"https://alimonitor.cern.ch/alihyperloop-data/trains/train.jsp?train_id={train_id}"
    try:
        return requests.get(
            url,
            verify=False,
            cert=(f"/tmp/tokencert_{os.getuid()}.pem", f"/tmp/tokenkey_{os.getuid()}.pem"),
            timeout=10,
        )
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        raise


def find_ao2ds(ali: alien.AliEn, aliendir: str) -> list[str]:
    """Find AO2Ds in train output directory"""
    cmd_find = f"find {PurePosixPath(aliendir) / 'AOD'} AO2D.root"
    print(cmd_find)
    ret = ali.run(cmd_find)
    if ret.exitcode != 0:
        print(f"Failed to run search: {cmd_find}\n{ret.out}")
        return []
    return ret.out.split()


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description="Download AO2Ds from hyperloop train")
    parser.add_argument("train_id", type=int, help="train ID")
    parser.add_argument("--prefix", "-p", default="/data2/MLhep/trains/")
    parser.add_argument("--dry-run", "-n", action="store_true", help="dry run")
    args = parser.parse_args()

    print("Obtaining train spec ..")
    train_spec = get_train_spec(args.train_id)
    outputdirs = [d["outputdir"] for d in train_spec.json()["jobResults"]]

    print("Finding AO2Ds ..")
    a = alien.AliEn()
    src = [d for outputdir in outputdirs for d in find_ao2ds(a, outputdir)]
    dst = ["file:" + str(PurePosixPath(args.prefix) / str(args.train_id) / file.lstrip("/")) for file in src]
    print("Files to copy:")
    for s, d in zip(src, dst):
        print(f"{s} -> {d}")

    if not args.dry_run:
        print("Copying ..")
        xrd_core.DO_XrootdCp(a.wb(), api_src=src, api_dst=dst)


if __name__ == "__main__":
    main()
