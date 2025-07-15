#!/bin/bash

DB_PATTERN="database_ml_parameters_LcToPKPi_multiclass_fdd" # Original database to be used as template
DB_DIR="data/data_run3"
OUT_DB_DIR="multitrial-db" # Directory to store multitrial databases only
ext=".yml"

DIR_PATH="/data8/majak/MLHEP"
DIR_PATTERN="results-24022025-newtrain-multitrial-prompt" # Prefix of output directory for fit results

# Paths to masshistos to fit
BASE_DIR="/data8/majak/MLHEP/results-24022025-newtrain-ptshape-prompt"
DATA_HIST="LHC23pp/Results/resultsdatatot/masshisto.root"
MC_HIST="LHC24pp_mc/Results/resultsmctot/masshisto.root"

# Run this only once to generate databases
# Then, you can comment this out if you don't change the *.py file
# The output analysis dir is set in databases to DIR_PATTERN + suffix with trial name
python run-mlhep-fitter-multitrial.py "${DB_PATTERN}" "${DB_DIR}" "${OUT_DB_DIR}" "${DIR_PATTERN}" || exit 1

for db in ${OUT_DB_DIR}/*.yml ; do
  db_basename=`basename ${db}`
  db_basename_no_ext=${db_basename%%${ext}}
  echo ${db_basename_no_ext}
  suffix=${db_basename_no_ext##${DB_PATTERN}}
  echo "suffix: ${suffix}"
  RESPATH="${DIR_PATH}/${DIR_PATTERN}${suffix}"
  echo "respath: ${RESPATH}"

  # Copy base masshistos so as to skip the masshisto step
  # Only the fit step needs to be activated in analyzer.yml
  # You need first to create the directory trees
  cp "${BASE_DIR}/${DATA_HIST}" "${RESPATH}/${DATA_HIST}"
  cp "${BASE_DIR}/${MC_HIST}" "${RESPATH}/${MC_HIST}"

  mlhep logfile_${db_basename}.log \
    -a Run3analysis \
    --run-config submission/analyzer.yml \
    --database-analysis ${db}

  # Copy the plots from MachineLearningHEP/machine_learning_hep/fig/ to RESPATH
  # It's not compulsory, it's just for you to see the fits
  # It might be obsolete if you changed the default output fig/ location in MLHEP
  rm -rf ${RESPATH}/fig/
  mv fig/ ${RESPATH}/fig/
done

