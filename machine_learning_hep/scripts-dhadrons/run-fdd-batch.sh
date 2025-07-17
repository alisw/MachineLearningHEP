#!/bin/bash

# Run MLHEP in batch for various non-prompt cuts
# You need a MLHEP database with %resdir%, %bkg...%, and %fd...% placeholders.

source "${HOME}/Run3Analysisvalidation/exec/utilities.sh"
WORKDIR="${HOME}/MachineLearningHEP/machine_learning_hep/"

# Base database.
DATABASE="database_ml_parameters_LcToPKPi_multiclass_fdd"
DATABASE_EXT="${DATABASE}.yml"
DATABASE_PATH="${WORKDIR}/data/data_run3/${DATABASE_EXT}"

# Output base directory to store all output subdirectories.
RESDIR="/data8/majak/MLHEP"

# Prefix of the output directories names.
#RESDIR_PATTERN="results-24022025-prompt"
RESDIR_PATTERN="results-24022025-newtrain-ptshape-prompt"

# Bkg cut. You can rewrite this to have different cuts in different pT bins.
bkg=0.00

# Loop over all non-prompt cuts.
for fd in $(seq 0.000 0.005 0.000) ; do
  echo "fd ${fd}"

  # Variable suffix to append to the output directory name.
  suffix="fd_${fd}"

  RESPATH="${RESDIR}/${RESDIR_PATTERN}${suffix}"

  CUR_DB="${DATABASE}_edit_fd${fd}.yml"
  cp "${DATABASE_PATH}" "${CUR_DB}" || ErrExit "Could not copy database"

  # Adjust the output directory
  sed -i "s/%resdir%/${RESDIR}/g" "${CUR_DB}" || ErrExit "Could not edit database"

  # Set bkg BDT cuts
  sed -i "s/%bkg01%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg12%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg23%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg34%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg45%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg56%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg67%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg78%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg810%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg1012%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg1216%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg1624%/${bkg}/g" "${CUR_DB}" || ErrExit "Could not edit database"

  # Set non-prompt BDT cuts
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"

  # `yes` is a program that says `y` to all interactive console prompts.
  # In this way, we skip all MLHEP questions about deleting old results.
  yes | mlhep --log-file "logfile_${suffix}.log" \
      -a Run3analysis \
      --run-config submission/analyzer.yml \
      --database-analysis "${CUR_DB}" \
      --delete \
     > "debug_${suffix}.txt" 2>&1 || ErrExit "Analysis failed"
done
