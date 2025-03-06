#!/bin/bash

# Compare ROOT files between two directories.

[ "$2" ] || { echo "Provide two paths to compare."; exit 1; }

for d in "$1" "$2"; do
    [ -d "$d" ] || { echo "Path $d is not a directory."; exit 1; }
done

dir_this="$(dirname "$(realpath "$0")")"
dir_pwd="$PWD"
dir_1="$(realpath "$1")"
dir_2="$(realpath "$2")"

shift
shift

readarray -t files < <(find "$dir_1" -maxdepth 3 -name "*.root")
[ ${#files[@]} -eq 0 ] && { echo "No ROOT files found in $dir_1."; exit 0; }

for f in "${files[@]}"; do
    file_1="$f"
    file_2="${file_1/$dir_1/$dir_2}"
    [ -f "$file_2" ] || { echo "File $file_2 does not exist. Skipping."; continue; }
    echo "Comparing $file_1 and $file_2"
    dir_out="${file_1/$dir_1\//}"
    dir_out="${dir_out/.root/}"
    echo "Output dir $dir_out"
    mkdir -p "$dir_out"
    cd "$dir_out" || { echo "Cannot enter $dir_out"; exit 1; }
    "${dir_this}/compare_root_files.py" "$file_1" "$file_2" "$@" > "diff.txt"
    cd "$dir_pwd" || { echo "Cannot enter $dir_pwd"; exit 1; }
done
