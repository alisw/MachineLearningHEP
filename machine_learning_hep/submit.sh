#!/bin/bash

##### Configuration

# Analysis stage

STAGE="all_off"
# STAGE="full_analysis"
# STAGE="preprocess"
# STAGE="data"
# STAGE="mc"
# STAGE="mltrain"
# STAGE="mlapp"
# STAGE="analysis"
# STAGE="processor"
# STAGE="analyzer"
# STAGE="variations"
# STAGE="systematics"
# STAGE="plotting"

# Suffix of the analysis database name

DATABASE="D0Jet_pp"
# DATABASE="LcJet_pp"

# Name of the analysis section in the analysis database

ANALYSIS="jet_obs"

##### Initialisation

DIR_THIS="$(dirname "$(realpath "$0")")"  # This directory
DBDIR="data/data_run3"
DB_DEFAULT="${DIR_THIS}/${DBDIR}/database_ml_parameters_${DATABASE}.yml"
LOG="log_${STAGE}.log"

##### Execution

echo "$(date) Start"
echo "Running the \"${STAGE}\" stage of the \"${ANALYSIS}\" analysis from the \"${DATABASE}\" database"

if [[ "${STAGE}" == "plotting" ]]; then
    echo "Log file: $LOG"
    python "${DIR_THIS}/plotting/plot_jetsubstructure_run3.py" -d "${DB_DEFAULT}" -a "${ANALYSIS}" > "${LOG}" 2>&1
elif [[ "${STAGE}" == "systematics" ]]; then
    echo "Log file: $LOG"
    python "${DIR_THIS}/analysis/do_systematics.py" -d "${DB_DEFAULT}" -a "${ANALYSIS}" > "${LOG}" 2>&1
elif [[ "${STAGE}" == "variations" ]]; then
    DB_VARIATION="${DIR_THIS}/${DBDIR}/database_variations_${DATABASE}_${ANALYSIS}.yml"
    CONFIG_FILE="${DIR_THIS}/submission/analysis.yml"
    "${DIR_THIS}/submit_variations.sh" "${DB_DEFAULT}" "${DB_VARIATION}" "${ANALYSIS}" "${CONFIG_FILE}"
else
    echo "Log file: $LOG"
    CONFIG_FILE="${DIR_THIS}/submission/${STAGE}.yml"
    CMD_ANA="mlhep -a ${ANALYSIS} -r ${CONFIG_FILE} -d ${DB_DEFAULT} -b --delete"
    ${CMD_ANA} > "${LOG}" 2>&1
fi || echo "Error"

echo "$(date) Done"
