#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Wrong number of parameters"
  exit 1
fi

DB=$1
LOGFILE=$2

mlhep --log-file ${LOGFILE} \
  --run-config submission/default_complete.yml \
  --database-analysis ${DB} \
  --delete
