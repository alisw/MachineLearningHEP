#!/bin/bash

[ "$4" ] || { echo "Usage: $0 <default database> <variation database> <analysis>"; exit 0; }

ErrExit() { echo "Error"; exit 1; }

# This directory
DIR_THIS="$(dirname "$(realpath "$0")")"

DB_DEFAULT="$1"
DB_VARIATION="$2"
ANALYSIS="$3"
CONFIG_FILE="$4"
RUN=0
CMD_VAR="python ${DIR_THIS}/do_variations.py ${DB_DEFAULT} ${DB_VARIATION}"
declare -a NJOBS  # number of parallel jobs
NJOBS[0]=50  # for variations without processor
NJOBS[1]=5   # for variations with processor
SCRIPT="script.sh" # name of the script with the execution lines

${CMD_VAR} || ErrExit

echo -e "\nDo you wish to run these variations?"
while true; do
  read -r -p "Answer: " yn
  case $yn in
    [y] ) echo "Proceeding"; RUN=1; break;;
    [n] ) echo "Aborting"; break;;
    * ) echo "Please answer y or n.";;
  esac
done

if ((RUN)); then
  echo -e "\nRunning variations"
  for PROC in 0 1; do
    ${CMD_VAR} -a "${ANALYSIS}" -r "${CONFIG_FILE}" -s "$SCRIPT" -p $PROC && parallel --will-cite --progress -j ${NJOBS[$PROC]} < "$SCRIPT"
  done || ErrExit
else
  echo -e "\nCleaning"
  ${CMD_VAR} -c -s "$SCRIPT" || ErrExit
fi
