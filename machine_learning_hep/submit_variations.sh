#!/bin/bash

[ "$3" ] || { echo "Usage: $0 <default database> <variation database> <analysis>"; exit 0; }

ErrExit() { echo "Error"; exit 1; }

DB_DEFAULT="$1"
DB_VARIATION="$2"
ANALYSIS="$3"
RUN=0
CMD_VAR="python do_variations.py ${DB_DEFAULT} ${DB_VARIATION}"
NJOBS=5 # number of parallel jobs
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
  { ${CMD_VAR} -a "${ANALYSIS}" -s "$SCRIPT" && parallel --will-cite --progress -j $NJOBS < "$SCRIPT"; } || ErrExit
else
  echo -e "\nCleaning"
  ${CMD_VAR} -c -s "$SCRIPT" || ErrExit
fi

exit 0
