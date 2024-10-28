#!/bin/bash

source "${HOME}/Run3Analysisvalidation/exec/utilities.sh"

WORKDIR="${HOME}/MachineLearningHEP/machine_learning_hep/"
DATABASE="${WORKDIR}/data/data_run3/database_ml_parameters_LcToPKPi_multiclass_ana_hyp_ml"
DATABASE_EXT="${DATABASE}.yml"

BKG_812=0.25
BKG_1224=0.30
for bkg in $(seq 0.20 0.05 0.5) ; do
  echo "bkg ${BKG_812} ${BKG_1224}"

  RESDIR_PATTERN="results-2207-hyp-ml_bkg"
  RESDIR="${RESDIR_PATTERN}_${BKG_812}_${BKG_1224}\/"
  RESPATH="/data8/majak/MLHEP/${RESDIR}"

  rm -rf "${RESPATH}"

  CUR_DB="${DATABASE}_edit_bkg${bkg}.yml"
  cp "${DATABASE_EXT}" "${CUR_DB}" || ErrExit "Could not copy database"

  sed -i "s/${RESDIR_PATTERN}.*/${RESDIR}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg812%/${BKG_812}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg1224%/${BKG_1224}/g" "${CUR_DB}" || ErrExit "Could not edit database"

  mlhep --log-file "logfile_bkg${bkg}.log" \
      --run-config submission/default_complete.yml \
      --database-analysis "${CUR_DB}" \
      --delete
done
