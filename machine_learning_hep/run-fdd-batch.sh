#!/bin/bash

source "${HOME}/Run3Analysisvalidation/exec/utilities.sh"

WORKDIR="${HOME}/MachineLearningHEP/machine_learning_hep/"
DATABASE="database_ml_parameters_LcToPKPi_multiclass_fdd"
DATABASE_EXT="${DATABASE}.yml"
DATABASE_PATH="${WORKDIR}/data/data_run3/${DATABASE_EXT}"
#RESDIR_PATTERN="results-24022025-prompt"
RESDIR_PATTERN="results-24022025-newtrain-ptshape-prompt"

bkg=0.00
for fd in $(seq 0.000 0.005 0.000) ; do
  echo "fd ${fd}"

  #suffix="fd_${fd}"
  suffix=""
  RESDIR="${RESDIR_PATTERN}${suffix}"
  RESPATH="/data8/majak/MLHEP/${RESDIR}/"

  #rm -rf "${RESPATH}"

  CUR_DB="${DATABASE}_edit_fd${fd}.yml"
  cp "${DATABASE_PATH}" "${CUR_DB}" || ErrExit "Could not copy database"

  sed -i "s/%resdir%/${RESDIR}/g" "${CUR_DB}" || ErrExit "Could not edit database"
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
  sed -i "s/%fd01%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
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

  yes | mlhep --log-file "logfile_${suffix}.log" \
      -a Run3analysis \
      --run-config submission/analyzer.yml \
      --database-analysis "${CUR_DB}" \
      --delete \
     > "debug_${suffix}.txt" 2>&1 || ErrExit "Analysis failed"
done
