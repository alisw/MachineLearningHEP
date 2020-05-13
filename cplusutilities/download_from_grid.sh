#!/bin/bash

# Script to download any files from the Grid

speed=10 # number of download threads started per second

# Check correct input.

if [ -z "$3" ]; then
    echo "Usage: $0 <Grid source path> <local target path> <file names>"
    exit 1
fi

# Check correct environment.

if [ -z "$ALIPHYSICS_RELEASE" ] || [ -z "$(echo $ALIPHYSICS_RELEASE | grep JALIEN)" ]; then
    echo "Error: Load the JALIEN flavour of AliPhysics and run the script again."
    echo '/cvmfs/alice.cern.ch/bin/alienv enter AliPhysics/vAN-'$(date --date="-2 days" +%Y%m%d)'_JALIEN-1'
    exit 1
fi

path_grid="$1" # Grid path
shift
path_local="$1" # Local path
shift
filenames="$@" # Names of files

# Create list of files.

timestamp=$(date +%Y%m%d_%H%M%S)_${BASHPID}
inputlist=filelist_${timestamp}.txt
logfile=stdouterr_${timestamp}.log
rm -f $inputlist $logfile
echo "Creating list of files"
for file in $filenames; do
    alien_find $path_grid/ $file >> $inputlist
    if [ ! $? -eq 0 ]; then echo "Error"; exit 1; fi
done
nfiles=$(cat $inputlist | wc -l)

# Display summary and ask for confirmation.

echo "Source Grid path: $path_grid"
echo "Target local path: $path_local"
echo "File names: $filenames"
echo "Number of files: $nfiles"

echo -e "\nDo you wish to continue? (y/n)"
while true; do
    read -p "Answer: " yn
    case $yn in
        [y] ) echo "Proceeding"; break;;
        [n] ) echo "Aborting"; rm -f $inputlist; exit 0; break;;
        * ) echo "Please answer y or n.";;
    esac
done

# Download.

delay=$(echo "scale=10 ; 1 / $speed" | bc)

i_file=0
for file in $(cat $inputlist); do
    i_file=$((i_file + 1))
    target_file="$path_local/${file/$path_grid/}"
    echo "$i_file/$nfiles $file"
    mkdir -p $(dirname $target_file)
    if [ ! $? -eq 0 ]; then echo "Error"; exit 1; fi
    path_alien="alien://${file}"
    alien_cp -f ${path_alien} ${target_file} >> $logfile 2>&1 &
    sleep $delay
done

# Watch progress.

nsuccess=0
done=0
pause=2
while [ $done -lt $nfiles ]; do
    nsuccess=$(cat $logfile | grep -e "MESSAGE: \[SUCCESS\]" -e "TARGET VALID" | wc -l)
    nerror=$(cat $logfile | grep "MESSAGE: \[ERROR\]" | wc -l)
    done=$((nsuccess + nerror))
    echo -e "Completed: $nsuccess/$nfiles\tFailed: $nerror/$nfiles\tDone: $done/$nfiles"
    if [ $done -lt $nfiles ]; then
        sleep $pause
    fi
done

# Clean.

rm -f $inputlist $logfile

exit 0

