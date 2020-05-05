#!/bin/bash

# Delete per-period result directories.

dir_root="$1"

if [[ -z "$dir_root" ]]
then
    echo "Error: Empty directory."
    exit 1
fi

if [[ ! -d "$dir_root" ]]
then
    echo "Error: Directory $dir_root does not exist."
    exit 1
fi

pattern="pp_201*"

dir_found=$(find $dir_root -type d -name $pattern)

if [[ -z "$dir_found" ]]
then
    echo "Nothing found."
    exit 0
fi

echo "Found these directories to delete:"
for d in $dir_found
do
    echo $d
done
echo "Do you wish to delete them?"
while true; do
    read -p "Answer: " yn
    case $yn in
        [Yy] ) echo "Deleting"
        for d in $dir_found
        do
            rm -rf $d
        done
        break;;
        [Nn] ) echo "Skipping"; break;;
        * ) echo "Please answer y or n.";;
    esac
done

echo "Done"

exit 0

