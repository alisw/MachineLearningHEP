import subprocess
import os
import errno
from shutil import rmtree
from tempfile import mkdtemp
from argparse import ArgumentParser

DEFAULT_DEST = "~/.machine_learning_hep/data/inputroot"
SOURCE = "https://www.dropbox.com/sh/a9zviv7fz0dv7co/AABMNfZWzxUFUd8VszbAwlSRa?dl=1"

def main():
    argp = ArgumentParser(description="Download or update input data for MachineLearningHEP")
    argp.add_argument("--verbose", dest="verbose", default=False, action="store_true",
                      help="Be verbose")
    argp.add_argument("--clean", dest="clean", default=False, action="store_true",
                      help="Remove old data before downloading")
    argp.add_argument("--dest", dest="dest", default=DEFAULT_DEST,
                      help=f"Where to download input data (defaults to {DEFAULT_DEST})")
    args = argp.parse_args()

    args.dest = os.path.expanduser(args.dest)

    # Create directory structure if not existing
    try:
        os.makedirs(args.dest)
    except OSError as exc:
        if not os.path.isdir(args.dest) or exc.errno != errno.EEXIST:
            print(f"Cannot create directory {args.dest}: check your permissions")
            exit(1)

    if args.verbose:
        redir = None
    else:
        redir = open(os.devnull, "w")

    # Download zip file to a temporary directory
    tmpDir = mkdtemp()
    zipFile = os.path.join(tmpDir, "data.zip")
    print("Downloading data...")
    try:
        subprocess.check_call(["curl", "-L", SOURCE, "-o", zipFile], stdout=redir, stderr=redir)
    except subprocess.CalledProcessError:
        print(f"Error downloading source data file {SOURCE}")
        exit(2)

    if args.clean:
        print("Removing old data...")
        try:
            rmtree(args.dest)
        except OSError as exc:
            print(f"Cannot remove old data under {args.dest}")
            exit(3)

    # Unpack data
    print("Unpacking...")
    try:
        subprocess.check_call(["unzip", "-o", "-d", args.dest, zipFile], stdout=redir, stderr=redir)
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 2:
            print(f"Error unpacking {zipFile} into {args.dest}: zip file has been kept")
            exit(4)

    # All OK: clean up
    try:
        rmtree(tmpDir)
    except OSError:
        pass

    print(f"All done: data downloaded under {args.dest}")
