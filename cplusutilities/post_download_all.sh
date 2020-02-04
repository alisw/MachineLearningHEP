#!/bin/bash

# Run the merging script for all "unmerged" directories in the input path.

path="$1"
if [ ! -d "$path" ]
then
  echo "Input path $path does not exist!"
  exit 1
fi

for dir in $(find $path -type d -name unmerged)
do
  dirout=${dir/unmerged/merged}
  if [ -d "$dirout" ]
  then
    echo "Output directory $dirout already exists. Skipping"
    continue
  fi
  ./post_download.sh --input ${dir/unmerged/} --target-size 500000 --jobs 50 -f
done

exit 0

