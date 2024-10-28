#!/bin/bash

source "${HOME}/Run3Analysisvalidation/exec/utilities.sh"

WORKDIR="${HOME}/MachineLearningHEP/machine_learning_hep/"
DATABASE="${WORKDIR}/data/data_run3/database_ml_parameters_LcToPKPi_multiclass_fdd"
DATABASE_EXT="${DATABASE}.yml"
RESDIR_PATTERN="results-2308-hyp-ml_1224_split_widerange_"

BKG_1216=0.60
BKG_1624=0.60

for fd in $(seq 0.00 0.01 0.00) ; do
  echo "bkg  ${BKG_1216} ${BKG_1624} fd ${fd}"

  RESDIR="${RESDIR_PATTERN}bkg_${BKG_1216}_${BKG_1624}_fd_${fd}"
  RESPATH="/data8/majak/MLHEP/${RESDIR}/"

  rm -rf "${RESPATH}"

  CUR_DB="${DATABASE}_edit_fd${fd}_bkg_${BKG_1216}_${BKG_1624}.yml"
  cp "${DATABASE_EXT}" "${CUR_DB}" || ErrExit "Could not copy database"

  sed -i "s/%resdir%/${RESDIR}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg1216%/${BKG_1216}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%bkg1624%/${BKG_1624}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd12%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd23%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd34%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd45%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd56%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd68%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd812%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd1216%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"
  sed -i "s/%fd1624%/${fd}/g" "${CUR_DB}" || ErrExit "Could not edit database"

  mlhep --log-file "logfile_fd${fd}_bkg_${BKG_1216}_${BKG_1624}.log" \
      --run-config submission/default_complete.yml \
      --database-analysis "${CUR_DB}" \
      --delete \
     > "debug_fd${fd}_bkg_${BKG_1216}_${BKG_1624}.txt" 2>&1 || ErrExit "Analysis failed"
done
