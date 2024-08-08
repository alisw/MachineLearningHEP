#!/bin/bash

# This directory
DIR_THIS="$(dirname "$(realpath "$0")")"

# Config file prefix
# CONFIG="default"
# CONFIG="d0jet"
CONFIG="lcjet"

# Config file suffix
# STAGE="complete"
# STAGE="all"
# STAGE="ana"
STAGE="variations"

# Suffix of the analysis database
DATABASE="LcJet_pp"

# Name of the analysis section in the analysis database
ANALYSIS="jet_obs"

DBDIR="data/data_run3"
DB_DEFAULT="${DIR_THIS}/${DBDIR}/database_ml_parameters_${DATABASE}.yml"

if [[ "${STAGE}" == "variations" ]]; then
    echo "Running the variation script for the ${ANALYSIS} analysis from ${DATABASE}"
    DB_VARIATION="${DIR_THIS}/${DBDIR}/database_variations_${DATABASE}_${ANALYSIS}.yml"
    "${DIR_THIS}/submit_variations.sh" "${DB_DEFAULT}" "${DB_VARIATION}" "${ANALYSIS}"
else
    CONFIG_FILE="${DIR_THIS}/submission/${CONFIG}_${STAGE}.yml"
    CMD_ANA="mlhep -a ${ANALYSIS} -r ${CONFIG_FILE} -d ${DB_DEFAULT} -c"
    echo "Running the \"${STAGE}\" stage of the \"${CONFIG}\" configuration of the \"${ANALYSIS}\" analysis from ${DATABASE}"
    ${CMD_ANA}
fi || { echo "Error"; exit 1; }

echo -e "\n$(date)"

# DIR_RESULTS="/data/DerivedResultsJets/D0kAnywithJets/vAN-20200304_ROOT6-1/"
# echo -e "\nCleaning ${DIR_RESULTS}"
# "${DIR_THIS}/clean_results.sh" "${DIR_RESULTS}"

echo -e "\nDone"
