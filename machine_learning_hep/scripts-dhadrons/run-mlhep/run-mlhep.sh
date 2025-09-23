#!/bin/bash

# Shortcut to run MLHEP
# Usage: ./run-mlhep.sh database_Lc.yml submission/analysis.yml logfile.log

if [ "$#" -ne 3 ]; then
  echo "Wrong number of parameters"
  exit 1
fi

DB=$1
CONFIG=$2
LOGFILE=$3

mlhep --log-file ${LOGFILE} \
  -a Run3analysis \
  --run-config ${CONFIG} \
  --database-analysis ${DB}
